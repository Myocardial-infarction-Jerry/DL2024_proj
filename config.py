from utils import get_weight_path, get_weight_list

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d", "wide_resnet50_2", "wide_resnet101_2",
           "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]

NET_NAME = 'resnet50'
VERSION = 'ResNet2.0'
DEVICE = '0'
# Must be True when pre-training and inference
PRE_TRAINED = True
# LOAD_MODEL = "pre_trained/ViT-B_16.pth"
LOAD_MODEL = None
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5

CUB_TRAIN_MEAN = [0.485, 0.456, 0.406]
CUB_TRAIN_STD = [0.229, 0.224, 0.225]

CKPT_PATH = './ckpt/{}/fold{}'.format(VERSION, CURRENT_FOLD)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
# print(WEIGHT_PATH)

if PRE_TRAINED:
    try:
        WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/'.format(VERSION))
    except:
        WEIGHT_PATH_LIST = None
else:
    if LOAD_MODEL:
        WEIGHT_PATH_LIST = LOAD_MODEL
    else:
        WEIGHT_PATH_LIST = None

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-4,
    'n_epoch': 1000,
    'num_classes': 200,
    'image_size': 224,
    'batch_size': 80,
    'train_mean': CUB_TRAIN_MEAN,
    'train_std': CUB_TRAIN_STD,
    'num_workers': 8,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'gamma': 0.1,
    'milestones': [100, 200, 300],
    'T_max': 5,
    'use_fp16': True,
    'dropout': 0.01
}

# Arguments when perform the trainer
SETUP_TRAINER = {
    'output_dir': './ckpt/{}'.format(VERSION),
    'log_dir': './log/{}'.format(VERSION),
    'optimizer': 'Adam',
    'loss_fun': 'Cross_Entropy',
    'class_weight': None,
    'lr_scheduler': 'MultiStepLR'
}
