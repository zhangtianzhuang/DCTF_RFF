from common_config import *
from util import *

train_point = common_config_train_point
test_point = common_config_test_point
train_snr = common_config_train_snr
test_snr = common_config_test_snr
aug = common_config_aug
slice_number = common_config_slice_number
slice_size = common_config_slice_size
devices = common_config_devices

data_type = 'Test'
root = '/mnt/DiskA-1.7T/ztz/WIFI_Dataset/AugData/' + common_config_train_set + \
       '/' + data_type + '/' + 'Slice' + str(slice_number) + '-' + str(slice_size)
target_dir = '/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF/class/' + common_config_train_set
global point
global snr
if data_type == 'Test':
    point = train_point
    snr = train_snr
    content = list()
    for p in point:
        for d in devices:
            for a in aug:
                for s in snr:
                    line = '{}/P{}_D{}_A{}_S{}_L{}-{}_{}.mat'.format(
                        root, p, d, a, s, slice_number,
                        slice_size, data_type)
                    content.append(line)
    # {地点}_{增扩}_{噪声}_L{切片数量}-{切片长度}
    target_file_name = '{}_{}_{}_L{}-{}'.format(buildP(point, 'TE'),
                            build_aug(aug), buildSNR(snr), slice_number, slice_size)
    target = target_dir + '/' + target_file_name + '.txt'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    write_list_to_file(content, target)
    print(target)
elif data_type == 'Test':
    point = test_point
    snr = test_snr
    for s in snr:
        content = list()
        for p in point:
            for d in devices:
                for a in aug:
                    line = '{}/P{}_D{}_A{}_S{}_L{}-{}_{}.mat'.format(
                        root, p, d, a, s, slice_number,
                        slice_size, data_type)
                    content.append(line)
        # {地点}_{增扩}_{噪声}_L{切片数量}-{切片长度}
        target_file_name = '{}_{}_{}_L{}-{}'.format(buildP(point, 'TE'),
                                    build_aug(aug), buildSNR([s]), slice_number, slice_size)
        target = target_dir + '/' + target_file_name + '.txt'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        write_list_to_file(content, target)
        print(target)
