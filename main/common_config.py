# DATASET Name
DATASET_ClearedDataset_20 = 'ClearedDataset-20'
DATASET_ClearedDataset_1 = 'ClearedDataset-1'
DATASET_Cleared1_ModelB_S1_10 = 'Cleared1_ModelB_S1-10'
DATASET_ClearedDataset_1_ModelB_S1_5 = 'ClearedDataset-1_ModelB_S1-5'
DATASET_ClearedDataset_20_ModelB_DS_S1_400 = 'ClearedDataset-20_ModelB_DS_S1-400'
DATASET_ClearedDataset_20_ModelB_DS_S401_500 = 'ClearedDataset-20_ModelB_DS_S401-500'
DATASET_ClearedDataset_20_ModelB_DS_S501_900 = 'ClearedDataset-20_ModelB_DS_S501-900'
DATASET_ClearedDataset_1_RawSlice = 'ClearedDataset-1-RawSlice'
OUTPUT_DIR = '/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF'

# Model Name
MODEL_M1 = 'M1'
MODEL_M2 = 'M2'
MODEL_M3 = 'M3'
MODEL_M4 = 'M4'
MODEL_M5 = 'M5'
MODEL_RESNET18 = 'resnet18'
MODEL_RESNET34 = 'resnet34'
MODEL_RESNET50 = 'resnet50'
MODEL_RESNET18_CUSTOM = 'resnet18-custom'

# Variable
common_config_model = MODEL_M5
common_config_train_set = DATASET_ClearedDataset_1_RawSlice
common_config_test_set = DATASET_ClearedDataset_1_RawSlice
common_config_devices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# cross training
common_config_train_point = ['1', '2']
common_config_test_point = ['1', '2']
common_config_train_snr = ['5', '10', '15', '20', '25', '30', 'no']
# common_config_train_snr = ['no']
common_config_test_snr = common_config_train_snr
common_config_device_count = len(common_config_devices)
common_config_learning_rate = 0.0001
common_config_batch_size = 1024
common_config_diff_channel = []
common_config_early_stopping_max_err_count = 10
common_config_aug = ['no']
common_config_slice_number = 10
common_config_slice_size = 128
