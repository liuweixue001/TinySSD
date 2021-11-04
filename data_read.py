import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l
# 读取数据集，并返回图像和标签
def read_data_bananas(is_train=True):
    """
    下载、解压数据集后放在py文件同级目录下即可
    http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip
    """
    data_dir = d2l.download_extract('./banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
        else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train
                else 'bananas_val', 'images', f'{img_name}')))
        targets.append(list(target))
    # print(torch.tensor(targets).unsqueeze(1)[0])
    return images, torch.tensor(targets).unsqueeze(1) / 256


# 自定义Dataset实例加载数据集
class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


# 加载实例
def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train = True),
                            batch_size = batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train = False),
                            batch_size = batch_size, shuffle=True)
    return train_iter, val_iter


if __name__ == '__main__':
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape)