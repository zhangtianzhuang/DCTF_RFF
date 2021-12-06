import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from MyDataset import MyDataset
import util
from common_config import *
import matplotlib.pyplot as plt


Model = common_config_model  # net model
epochs = 120
lr = common_config_learning_rate  # learning rate
# using the second GPU in the linux system, more info: command "gpustat"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

net = util.get_net(Model)
net = net.cuda()
epochs_name = str(epochs)
batch_size = common_config_batch_size
device_count = common_config_device_count  # the number of devices
print('epochs:', epochs, 'lr:', lr)
print('net create success:', net)

train_data_set_type = common_config_train_set
test_data_set_type = common_config_test_set
P = common_config_train_point
testP = common_config_test_point
snr = common_config_train_snr
test_snr = common_config_test_snr
snr_name_train = util.buildSNR(snr)
snr_name_test = util.buildSNR(test_snr)
c_name = util.buildC(device_count)
result = list()


# test
def test(para):
    # loading the parameters of the net model
    with torch.no_grad():
        path = OUTPUT_DIR + '/model/' + train_data_set_type + '/' + para
        pretrained_dict = torch.load(path)['state_dict']
    try:
        net.load_state_dict(pretrained_dict)
    except IOError:
        raise IOError("net weights not found")
    net.eval()
    test_correct = 0
    test_total = 0
    confuse_matrix = np.zeros([device_count, device_count], dtype=np.int)
    err_pos = []
    for i, data in enumerate(test_loader, 0):
        inputs, test_labels = data
        inputs, labels = Variable(inputs), Variable(test_labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
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
        # print('batch', i)
    result.append(round(100. * test_correct / test_total, 4))
    print('the number of err:', len(err_pos))
    print(err_pos)
    print("test acc: %.4f" % (100. * test_correct / test_total))
    names = [
        'device-1', 'device-2', 'device-3', 'device-4', 'device-5', 'device-6', 'device-7',
        'device-8', 'device-9', 'device-10'
    ]
    confuse_matrix_dir = OUTPUT_DIR + '/confuse_matrix/' + test_data_set_type
    util.plot_confusion_matrix(np.array(confuse_matrix), names[:device_count], dir_name=confuse_matrix_dir,
                               file_name=cm_name)


def build_para_name():
    return Model + '_' + train_common + '_BS_' + str(common_config_batch_size) + '_LR' + util.buildLR(lr) + '.pth'


if __name__ == "__main__":
    for tmp_snr in common_config_train_snr:
        snr_name_test = util.buildSNR([tmp_snr])
        # config
        train_common = train_data_set_type + '_' + util.buildP(P, 'TR') + '_' + c_name + '_' + snr_name_train
        # 训练后的模型参数命名
        para = build_para_name()
        txt_dir = OUTPUT_DIR + '/class' + '/' + test_data_set_type + '/'
        train_set_file = txt_dir + train_common + '.txt'
        test_set_file = txt_dir + test_data_set_type + '_' + util.buildP(testP, 'TE') + \
                        '_' + c_name + '_' + snr_name_test + '.txt'
        cm_name = Model + '_' + c_name + '_' + util.buildP(P, 'TR') + '_TR' + util.buildSNR(snr) \
                  + '_' + util.buildP(testP, 'TE') + '_TE_' + snr_name_test + '_BS' + str(common_config_batch_size) \
                  + '_LR' + util.buildLR(lr)
        print('Model name is', para)
        print('Test filename', test_set_file)
        print('CM Name', cm_name)
        # 第一步：数据加载与处理,选择训练设备cpu/gpu
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("是否使用GPU加速：", torch.cuda.is_available())
        # 为当前GPU/CPU设置随机种子(第三行是否必要存疑)
        test_data = MyDataset(txt=test_set_file, type='test', device_count=device_count, shuffle=False,
                              transform=transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()]))
        print('the type of test_data is', type(test_data))
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)
        print('创建DataLoader成功!')
        print('Parameters:', '\nDataType:', train_common, '\nBatchSize:', batch_size)
        test(para)
        print('epochs:', epochs, 'lr:', lr)
    print(result)
    plt.figure('Identification Accuracy')
    x = [i for i in range(5, 40, 5)]
    plt.xlabel('SNR(dB)')
    plt.ylabel('Accuracy(%)')
    plt.grid(True)
    plt.plot(x, result, 'r-o')
    plt.show()
