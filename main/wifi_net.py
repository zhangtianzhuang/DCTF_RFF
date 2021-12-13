import torch.nn as nn
from common_config import *
import torch


# 第二步：创建网络模型
class M1(nn.Module):
    def __init__(self):
        self.class_name = 'M1'
        super(M1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=16 * 13 * 13, out_features=256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=64)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=64, out_features=9)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv2_output = conv2_output.view(-1, 16 * 13 * 13)
        fc1_output = self.fc1(conv2_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        return fc3_output


class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.class_name = 'M2'
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=60, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=60 * 23 * 23, out_features=2 ** 13)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=2 ** 13, out_features=2 ** 12)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=2 ** 12, out_features=2 ** 11)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=2 ** 11, out_features=2 ** 10)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(in_features=2 ** 10, out_features=10)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1, 60 * 23 * 23)

        fc1_output = self.fc1(conv5_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        fc4_output = self.fc4(fc3_output)
        fc5_output = self.fc5(fc4_output)
        return fc5_output


class M3(nn.Module):
    def __init__(self):
        self.class_name = 'M3'
        super(M3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=(len(common_config_diff_channel) + 1) * 3, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=14 * 14 * 64, out_features=2 ** 12)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=2 ** 12, out_features=2 ** 10)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=2 ** 10, out_features=common_config_device_count)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        last_output = conv3_output.view(-1, 14 * 14 * 64)
        fc1_output = self.fc1(last_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        return fc3_output


class M4(nn.Module):
    def __init__(self):
        self.class_name = 'M4'
        super(M4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=(len(common_config_diff_channel) + 1) * 3, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=14 * 14 * 64, out_features=2 ** 11)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=2 ** 11, out_features=2 ** 9)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=2 ** 9, out_features=9)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        last_output = conv3_output.view(-1, 14 * 14 * 64)
        fc1_output = self.fc1(last_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        return fc3_output


class M5(nn.Module):
    def __init__(self):
        self.class_name = 'M5'
        super(M5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(1, 7)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5)),
            nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 7)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5)),
            nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 1))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128 * 64, out_features=128)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv2(conv2_output)
        conv4_output = self.conv2(conv3_output)
        last_output = conv4_output.view(-1, 1 * 128 * 64)
        fc1_output = self.fc1(last_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output


if __name__ == '__main__':
    # conv_1 = nn.Sequential(
    #         nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(1, 7)),
    #         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5)),
    #         nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
    #     )
    # conv_2 = nn.Sequential(
    #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 7)),
    #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5)),
    #     nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
    # )
    # tensor_1 = torch.randn(1, 2, 1, 1000)
    # output = conv_1(tensor_1)
    # print(output)
    # print(output.shape)
    # output_2 = conv_2(output)
    # print(output_2)
    # print(output_2.shape)
    pass
