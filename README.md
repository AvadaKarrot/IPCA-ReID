# Code Usage
[Framework](TransReID\figs\IPCA_REID.png)
![Framework](TransReID\figs\IPCA_REID.png)

## Dataset

To download and use the datasets, please refer to:  
- [**Market1501**  ](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ)
- [**MSMT17**](https://arxiv.org/abs/1711.08565)
- [**CUHK03**](https://drive.google.com/file/d/0B7TOZKXmIjU3OUhfd3BPaVRHZVE/view?resourcekey=0-hU4gyE6hFsBgizIh9DFqtA)

Follow the instructions from each dataset's official source to download the required files.  
Please organize the datasets with the following directory structures:

### Directory Structures


### Installation

Initially，please make sure your environment containing the required packages. if not, you can run:

```bash
$ cd repository
$ pip install -r requirements.txt
```

Then activate environment using:
```bash
$ conda activate env_name
```

# train_caption.py
Image Encoder + Text Encoder (CLIP)


- COOP.COOP_PROMPT: 可学习Prompt：ctx
- COCOOP.COCOOP_PROMPT :  meta block for each instance
- MAPLE.MAPLE_PROMPT: 多模态可学习Prompt: 文本-图像 (1-12 layer)
- Origin: "a photo of"
