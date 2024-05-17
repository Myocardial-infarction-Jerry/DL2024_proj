import torch
import numpy as np


def convert_npz_to_pth(npz_path, pth_path):
    # 加载 npz 文件
    npz_weights = np.load(npz_path)
    state_dict = {}

    # 遍历 npz 文件中的所有项
    for key, value in npz_weights.items():
        # 转换 numpy 数组为 torch tensor
        tensor = torch.from_numpy(value)
        state_dict[key] = tensor

    # 保存转换后的 state_dict 为 pth 文件
    torch.save(state_dict, pth_path)


if __name__ == "__main__":
    npz_path = 'pre_trained/ViT-B_16.npz'
    pth_path = 'pre_trained/ViT-B_16.pth'

    convert_npz_to_pth(npz_path, pth_path)
