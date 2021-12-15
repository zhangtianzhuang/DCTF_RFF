from util import *

# train_point = common_config_train_point
# test_point = common_config_test_point
# train_snr = common_config_train_snr
# test_snr = common_config_test_snr
# aug = common_config_aug
# slice_number = common_config_slice_number
# slice_size = common_config_slice_size
# devices = common_config_devices

"""
选择生成数据集的类型
1.生成所有的训练集
2.生成所有的测试集
3.按照SNR生成测试集
存储格式：
目录：{OUTPUT_DIR}/class/{数据集名称}
文件：{数据集类型}_{地点}_{增扩}_{噪声}_L{切片数量}-{切片长度}
参数：数据集名称，数据集类型，采样地点，增扩，噪声，切片数量，切片长度，是否按照SNR分开存储为单个文件
"""

# 1.获取需要的参数
train_para = get_train_parameter()
test_para = get_test_parameter()
current_filename = os.path.basename(__file__)[:-3]
logger = build_logger(current_filename)


def make_dataset(dp: DatasetParameter, is_divided_by_snr):
    slice_size = dp.slice_size
    slice_number = dp.slice_number
    point = dp.point
    snr = dp.snr
    aug = dp.aug
    name = dp.name
    type = dp.type
    device = dp.device
    root = '/mnt/DiskA-1.7T/ztz/WIFI_Dataset/AugData/{0}/{1}/Slice{2}-{3}' \
        .format(name, type, slice_number, slice_size)
    target_dir = '/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF/class/{}'.format(name)
    target_path = build_dataset_file_2(dp)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    content = dict()
    for s in snr:
        sub_content = list()
        for p in point:
            for d in device:
                for a in aug:
                    line = '{0}/P{1}_D{2}_A{3}_S{4}_L{5}-{6}_{7}.mat'.format(
                        root, p, d, a, s, slice_number, slice_size, type)
                    sub_content.append(line)
        content[str(s)] = sub_content
    if is_divided_by_snr:
        for s in snr:
            data = content[str(s)]
            target_path = build_dataset_file_2(dp, snr=s)
            write_list_to_file(data, target_path)
            logger.info('数据集写入成功：{}'.format(target_path))
    else:
        data = list()
        for s in snr:
            data.extend(content[str(s)])
        write_list_to_file(data, target_path)
        logger.info('数据集写入成功：{}'.format(target_path))


if __name__ == '__main__':
    make_dataset(train_para, is_divided_by_snr=False)
    make_dataset(test_para, is_divided_by_snr=False)
    make_dataset(test_para, is_divided_by_snr=True)
