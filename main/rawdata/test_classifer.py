import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from dataset import MyDataset, RawWiFiDataset
import util
from common_config import *
import matplotlib.pyplot as plt
from util import *

# 环境
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定（第2块）GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 网络参数
net = util.get_net(common_config_model).to(device)
batch_size, device_count = common_config_batch_size, common_config_device_count
train_data_set_type, test_data_set_type = common_config_train_set, common_config_test_set
train_point, test_point = common_config_train_point, common_config_test_point
train_snr, test_snr = common_config_train_snr, common_config_test_snr
# 结果
result = list()
# 模型的存储路径和文件名
net_model_para = common_config_net_model_para
model_dir = '{0}/model/{1}'.format(OUTPUT_DIR, train_data_set_type)
model_filename = build_model_para_name_2(net_model_para)
# 混淆矩阵的存储路径和文件名
# OUTPUT_DIR + '/confuse_matrix/' + test_data_set_type
confuse_matrix_dir = '{}/confuse_matrix/{}'.format(OUTPUT_DIR, train_data_set_type)
confuse_matrix_filename = build_confuse_matrix_name(build_model_para_name_2(net_model_para),
                                                    build_point(test_point), build_snr(test_snr))


# test
def test():
    # 1.加载训练后的网络模型
    with torch.no_grad():
        net_model = load_net_model_para(model_dir, model_filename)
    try:
        net.load_state_dict(net_model)
    except IOError:
        raise IOError("net weights not found")
    # 2.设置为训练模式，不会反向转播
    net.eval()
    test_correct, test_total = 0, 0
    confuse_matrix = np.zeros([device_count, device_count], dtype=np.int)
    err_pos = []  # 记录预测错误的详情，使用元组展示（样本位置，真实值，预测值）
    for i, data in enumerate(test_loader, 0):
        inputs, test_labels = data
        inputs = inputs.type(torch.FloatTensor)
        inputs, labels = Variable(inputs), Variable(test_labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, test_predicted = torch.max(outputs.data, 1)
        for idx, val in enumerate(labels.data):
            predicted_value = test_predicted[idx].cpu().numpy()
            real_value = val.cpu().numpy()
            confuse_matrix[real_value, predicted_value] += 1
            if predicted_value == real_value:
                test_correct += 1
            else:
                err_pos.append((i * batch_size + idx + 1, int(real_value) + 1,
                                int(predicted_value + 1)))
        test_total += test_labels.size(0)
    result.append(round(100. * test_correct / test_total, 4))
    print("test acc: %.4f" % (100. * test_correct / test_total))
    names = [
        'device-1', 'device-2', 'device-3', 'device-4', 'device-5', 'device-6', 'device-7',
        'device-8', 'device-9', 'device-10'
    ]
    util.plot_confusion_matrix(np.array(confuse_matrix), names[:device_count], dir_name=confuse_matrix_dir,
                               file_name=confuse_matrix_filename)


if __name__ == "__main__":
    for tmp_snr in common_config_train_snr:
        # 加载测试集
        test_file = build_dataset_file_2(get_test_parameter(), snr=tmp_snr)
        test_file_list = util.read_list_from_file(test_file)
        test_data = RawWiFiDataset(mat_file=test_file_list, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
        # 测试
        test()
    print(result)
    # 使用折线图展示结果
    if len(result) > 1:
        plt.figure('Identification Accuracy')
        x = [i for i in range(5, 40, 5)]
        plt.xlabel('SNR(dB)')
        plt.ylabel('Accuracy(%)')
        plt.grid(True)
        plt.plot(x, result, 'r-o')
        plt.show()
