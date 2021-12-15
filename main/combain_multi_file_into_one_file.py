from common_config import *
import util


def merge(file_list: list, target_file_name: str):
    with open(target_file_name, 'a+', encoding='utf-8') as target_file:
        # 清空目标文件
        target_file.truncate(0)
        # 将一些文件内容依次写入目标文件
        for file_name in file_list:
            for line in open(file_name):
                target_file.writelines(line)


if __name__ == '__main__':
    data_for = 'TE'
    P = ['2']
    # SNRs = ['5', '10', '15', '20', '25', '30', 'no']
    SNRs = ['10', '15', '20', '25', '30', 'no']
    snr_name = util.build_snr(SNRs)
    c_name = util.build_device_count(common_config_device_count)
    data_set_type = common_config_train_set if data_for == 'TR' else common_config_test_set
    root = '/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF/class/' + data_set_type
    target_file = root + '/' + data_set_type + '_' + util.build_point(P, data_for) + '_' \
                  + c_name + '_' + snr_name + '.txt'
    file_list = list()
    for s in SNRs:
        test_set_file = data_set_type + '_' + util.build_point(P, data_for) + '_' \
                        + c_name + '_' + util.build_snr([s]) + '.txt'
        file_list.append(root + '/' + test_set_file)
    print(target_file)
    print(file_list)
    merge(file_list, target_file)
