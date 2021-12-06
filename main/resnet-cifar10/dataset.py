import random
from torch.utils.data import Dataset
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class WiFiDataset(Dataset):
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
