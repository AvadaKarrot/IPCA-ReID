from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import ttach as tta

from utils.pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from utils.pytorch_grad_cam.utils.image import scale_cam_image
from utils.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils.pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class BaseCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers 

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        if tta_transforms is None:
            self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
        else:
            self.tta_transforms = tta_transforms

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        raise Exception("Not Implemented")

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations # bs, c, h, w
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1) # bs, h, w
        return cam

    def forward(
        self, input_tensor: torch.Tensor, 
        targets: List[torch.nn.Module],
        other_inputs=None, ## 可选的其他输入，例如文本描述等
        eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        # 如果有其他输入，也将它们移动到设备上（如果它们是张量）
        other_inputs = other_inputs.to(self.device) if isinstance(other_inputs, torch.Tensor) else other_inputs

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)


        self.outputs = outputs = self.activations_and_grads(input_tensor, other_inputs=other_inputs)

        # if targets is None:
        #     # target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        #     # targets = [ClassifierOutputTarget(category) for category in target_categories]
        #     targets = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        #     print(f'category id: {targets}')
        # else:
        #     assert len(targets) == input_tensor.size(0)

        # if self.uses_gradients:
        #     self.model.zero_grad()
        #     # loss = sum([target(output) for target, output in zip(targets, outputs)])
        #     loss = self.get_loss(outputs, targets)
        #     loss.backward(retain_graph=True)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)
        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            return depth, width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations] # bs dim h*w
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self, input_tensor: torch.Tensor, 
        targets: List[torch.nn.Module],
        other_inputs=None, ## 其他可选的输入，如文本描述等 
        eigen_smooth: bool = False
    ) -> np.ndarray:
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,  targets, eigen_smooth,other_inputs=other_inputs)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,  # 主要输入
        other_inputs=None,  # 可选的其他输入，支持多个
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth:
            return self.forward_augmentation_smoothing(input_tensor, targets=targets,other_inputs=other_inputs,  eigen_smooth=eigen_smooth)

        # 处理单输入和多输入情况
        if not other_inputs:  # 如果没有额外输入，处理单输入
            return self.forward(input_tensor, targets=targets, eigen_smooth=eigen_smooth)
        else:
            # 处理多输入情况，传递所有输入
            return self.forward(input_tensor, targets=targets, other_inputs=other_inputs, eigen_smooth=eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
