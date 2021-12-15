'''
动态配置
'''
from common_config import *


class CommonConfigPara:
    def __init__(self, model, train_set, test_set, train_point, test_point,
                 train_snr, test_snr, devices, learning_rate, batch_size,
                 early_stop_max_count, aug, slice_number=None, slice_size=None):
        self.aug = aug
        self.slice_size = slice_size
        self.slice_number = slice_number
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.devices = devices
        self.test_snr = test_snr
        self.train_snr = train_snr
        self.test_point = test_point
        self.train_point = train_point
        self.test_set = test_set
        self.train_set = train_set
        self.model = model
        self.early_stop_max_count = early_stop_max_count


def get_common_config_para_1() -> CommonConfigPara:
    return CommonConfigPara(common_config_model, common_config_train_set, common_config_test_set,
                            common_config_train_point, common_config_test_point,
                            common_config_train_snr, common_config_test_snr,
                            common_config_devices, common_config_learning_rate,
                            common_config_batch_size, common_config_early_stopping_max_err_count,
                            common_config_aug, common_config_slice_number,
                            common_config_slice_size)
