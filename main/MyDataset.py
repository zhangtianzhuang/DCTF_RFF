import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as scio
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, type='train', device_count=0, transform=None, target_transform=None,
                 loader=default_loader, shuffle=True):
        imgs = []
        # 按顺序读取文件，存入list中
        fh = open(txt, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.strip().split(' ')
            imgs.append((line[0], int(line[1]) - 1))

        total = [0 for i in range(0, device_count)]
        # 统计不同类别的样本数量
        for i, data in enumerate(imgs, 0):
            _, label = data
            total[label] = total[label] + 1

        print('数据集类型:', type)
        print('数据集大小：', len(imgs))
        print('设备1~' + str(device_count) + '数量分别为:', total)
        # 如有需要，将数据集打乱
        if shuffle:
            random.shuffle(imgs)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.total = total

    # get_item类似于一次次的取出数据
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)  # 将指定路径的图片信息加载进来
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class RawWiFiDataset(Dataset):
    def __init__(self, mat_file, shuffle=True):
        data = []
        mat_data = scio.loadmat(mat_file)
        waveform = mat_data['Store_Waveform']
        label = mat_data['Store_Frame_Label']
        # 遍历waveform的每一行，将数据转化为numpy.ndarray，标签存储到data中
        row, col = waveform.shape
        for i in range(0, row):
            line = waveform[i, :]
            real_part, imag_part = line.real.reshape(1, col), line.imag.reshape(1, col)
            tensor_data = np.stack((real_part, imag_part))  # 数据
            device_id = label[i, 0]  # 标签
            data.append((tensor_data, device_id))
        # 如有需要，将数据集打乱
        if shuffle:
            random.shuffle(data)
        self.data = data

    def __getitem__(self, index):
        waveform, label = self.data[index]
        return waveform, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # 加载mat文件，以及numpy的基本用法
    # mat_data = scio.loadmat('/mnt/DiskA-1.7T/ztz/test/P1-Slice.mat')
    # waveform = mat_data['Store_Waveform']
    # label: np.ndarray = mat_data['Store_Frame_Label']
    # print(type(waveform))
    # print(len(waveform))
    # print(waveform)
    # print(label)
    # print('first element is %d' % label[23][0])
    # print('label\'s dtype is', waveform.dtype)
    # print('label\'s shape is', label.shape)
    # print('label\'s ndim is', label.ndim)
    # print('the first row of waveform is', waveform[0, 0:9])
    # signal: np.ndarray = waveform[0, 0:5]
    # r: np.ndarray = signal.real
    # i: np.ndarray = signal.imag
    # print('real part:', r)
    # print('imag part:', i)
    # r2 = r.reshape(1, 5)
    # i2 = i.reshape(1, 5)
    # print('r2:', r2)
    # print('i2:', i2)
    # c = np.stack((r2, i2))
    # print('c:', c)
    # print(c.shape)

    # 测试RawWiFiDataset
    train_data = RawWiFiDataset(mat_file='/mnt/DiskA-1.7T/ztz/test/P1-Slice.mat',
                                   shuffle=False)
    train_loader = DataLoader(train_data, batch_size=2, num_workers=2, shuffle=True)
    for i, data in enumerate(train_loader, 0):
        wave, label = data
        print('wave', wave)
        print('wave shape is', wave.shape)
        print('label', label)
        break
