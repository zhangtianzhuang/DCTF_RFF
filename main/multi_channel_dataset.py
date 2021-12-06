import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch


def default_loader(path):
    return Image.open(path).convert('RGB')  # 不明白有什么作用？？？涉及数据加载问题


def get_other_channel(diff_channel: list, line: str):
    name = line.strip().split('/')
    res = []
    for d in diff_channel:
        name[-4] = 'Diff_' + str(d)
        name_1 = name[-1].split('_')
        name_1[-1] = 'I' + str(d) + '.jpg'
        name[-1] = '_'.join(name_1)
        res.append('/'.join(name))
    return res


class MultiChannelDataset(Dataset):
    def __init__(self, label_file='', type='train', diff_channel=[1], device_count=0, transform=None,
                 target_transform=None, loader=default_loader, shuffle=True):
        imgs = []
        # 按顺序读取文件，存入list中
        fh = open(label_file, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.strip().split(' ')
            other_diff_img = get_other_channel(diff_channel, line[0])
            sample = [int(line[1]) - 1, line[0]]
            res = sample + other_diff_img
            imgs.append(res)

        total = [0 for i in range(0, device_count)]
        # 统计不同类别的样本数量
        for i, data in enumerate(imgs, 0):
            label = data[0]
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
        self.diff_channel = diff_channel

    # get_item类似于一次次的取出数据
    def __getitem__(self, index):
        data = self.imgs[index]
        label = data[0]
        img_paths = data[1:]
        img_set = []
        for path in img_paths:
            img_set.append(self.loader(path))
        img_tensors = []
        if self.transform is not None:
            for s in img_set:
                img_tensors.append(self.transform(s))
        rest_tensors = img_tensors[1:]
        res_tensor = img_tensors[0]
        for t in rest_tensors:
            res_tensor = torch.cat((res_tensor, t), dim=0)
        return res_tensor, label
        # fn, label = self.imgs[index]
        # img = self.loader(fn)  # 将指定路径的图片信息加载进来
        # if self.transform is not None:
        #     img = self.transform(img)
        # return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # line = 'D:/WIFI_Dataset/DCTF_Image/MultiChannelClearedDataset-20/snr_no/Diff_1/P1/1/D1_P1-73.mat_Pos2_P1_Sno_I1.jpg'
    # res = get_other_channel([3, 5], line)
    # print(res)
    x1 = torch.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    x2 = torch.Tensor([[[13, 14, 15], [23, 23, 23]], [[23, 23, 23], [10, 11, 12]]])
    c1 = torch.cat((x1, x2), dim=0)
    print(c1)
