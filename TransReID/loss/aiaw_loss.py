import torch
import torch.nn as nn


class AIAWLoss(nn.Module):
    def __init__(self,reduction=None):
        super(AIAWLoss, self).__init__()
        self.reduction = reduction

    def forward(self, f_map, eye, mask_matrix, margin, num_remove_cov):
        '''
         Input:
            f_map_real (Tensor): The feature map for real data, with shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width of the feature map.
            eye_real (Tensor): The identity matrix used for real data.
            mask_matrix_real (Tensor): The mask matrix for real data, used for selective covariance calculation.

            margin_real (Tensor): Margin for real data.
            num_remove_cov_real (Tensor): The number of covariances to be removed for real data.
         Return:
             loss (Tensor): The AIAW loss
         '''
        f_cov, B = get_covariance_matrix(f_map, eye)
        f_cov_masked = f_cov * mask_matrix
        # 检查 num_remove_cov 是否为 0
        if num_remove_cov == 0:
            # print("警告：num_remove_cov 的值为 0，将设置为 1。")
            num_remove_cov = 1
        off_diag_sum = torch.sum(torch.abs(f_cov_masked), dim=(1,2), keepdim=True) - margin # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
        loss= torch.sum(loss) / B

        return loss


def get_covariance_matrix(f_map, eye=None):
    '''
     Input:
        f_map (Tensor): The feature map tensor with shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width of the feature map.
        eye (Tensor, optional): The identity matrix used for covariance calculation. Defaults to None.
     Return:
         f_cor (Tensor): The covariance matrix of the feature map
         B (int): batch size
     '''
    eps = 1e-5
    B, Seq, C = f_map.shape  # i-th feature size (B X C X H X W)

    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, -1, C)  # B
    f_cor = torch.bmm(f_map.transpose(1, 2), f_map ).div(Seq-1) + (eps * eye)  # C X C / HW

    return f_cor, B