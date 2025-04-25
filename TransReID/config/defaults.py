from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.L2_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False


# AdaIN Parameter
_C.MODEL.ADAIN = False
_C.MODEL.ADAIN_XISHU = (0.8, 1.2)
_C.MODEL.CSA = False
_C.MODEL.CSA_STYLE_NUM = 256

#### WHITE
_C.MODEL.WHITE = False 

############# clip prompts 
_C.MODEL.PROMPT = False
_C.MODEL.AUG_PRO = 0.5
_C.MODEL.AUG_LAYERS = 2

################ Real-Time Caption
_C.MODEL.CAPTION = False

############### text projection after clip
_C.MODEL.TEXT_PROJ = False

############### 文本图像特征融合的方法 'concat' 'ca'
_C.MODEL.FUSION = '' 

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# Random probability for Sobel filter:alpha*original+(1-alpha)*sobel 叠加sobel算子提取的边缘特征
_C.INPUT.SOBEL_PROB = 0.7
# 使用CLIP图像特征提取器，需要更改img_resolution
_C.INPUT.PERSON_REID = False
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.SOURCES = ['market1501']
_C.DATASETS.SOURCES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
_C.DATASETS.TARGETS = ('msmt17')
_C.DATASETS.TRANSFORMS = ['random_flip', 'pad','random_crop', 'random_erase']

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'RandomSampler'
# Number of instance for- one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_LR_INIT = 0.01
_C.SOLVER.LR_MIN = 0.000016
_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.RESUME_TRAIN = False

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH  = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# Grad-cam visulization: 与分类任务不同，测试不输出分类器结果，所以GRADCAM时候需要专门输出分类结果
_C.TEST.GRADCAM = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""

#  HDRNET
# -----------------------------------------------------------------------------
_C.HDRNET = CN()
_C.HDRNET.HDR_NET = False
_C.HDRNET.LUMA_BINS = 8
_C.HDRNET.CHANNEL_MULTIPLIER = 1
_C.HDRNET.SPATIAL_BIN = 16
_C.HDRNET.GUIDE_COMPLEXITY = 16
_C.HDRNET.BATCH_NORM = True
_C.HDRNET.NET_INPUT_SIZE = [256, 256]
_C.HDRNET.NET_OUTPUT_SIZE = 512
_C.HDRNET.NET_EMBEDDING = 256
_C.HDRNET.PRETRAIN_PATH = ''

########## text feature
_C.CLIP = CN()
_C.CLIP.TEXT_FEAT = False
_C.CLIP.CAPTION = False
_C.CLIP.TEXT_FEAT_PATH = None
_C.CLIP.TEXT_FEAT_QUERY_PATH = None
_C.CLIP.PROMPT = [] # 存放固定caption

############ Prompt Learner
_C.COOP = CN()
_C.COOP.COOP_PROMPT = False
_C.COOP.N_CTX = 16 
_C.COOP.CTX_INIT = ""
_C.COOP.CLASS_TOKEN_POSITION = 'end'

############ Conditional Prompt Learner
_C.COCOOP = CN()
_C.COCOOP.COCOOP_PROMPT = False
_C.COCOOP.N_CTX = 16 
_C.COCOOP.CTX_INIT = ""
_C.COCOOP.CLASS_TOKEN_POSITION = 'end'

########### Multiple MAPLE
_C.MAPLE = CN()
_C.MAPLE.MAPLE_PROMPT = False
_C.MAPLE.PROMPT_DEPTH = 9
_C.MAPLE.N_CTX = 3
_C.MAPLE.CTX_INIT = "a photo of"
_C.MAPLE.PREC = 'fp16'

################### Real-Time Caption
_C.CAPTION = CN()
_C.CAPTION.CAP_NUM = [0]

############## 频域网络 GFNet #############
_C.GFNET = CN()
_C.GFNET.PATCH_SIZE = 4
_C.GFNET.EMBED_DIM = [768]
_C.GFNET.DEPTH = [12]
_C.GFNET.MLP_RATIO = [4]
_C.GFNET.DROP_PATH = 0.1