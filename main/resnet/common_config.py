# DATASET Name
DATASET_ClearedDataset_20 = 'ClearedDataset-20'
DATASET_ClearedDataset_1 = 'ClearedDataset-1'
DATASET_ClearedDataset2_1 = 'ClearedDataset2-1'
DATASET_ClearedDataset2_1_Diff10 = 'ClearedDataset2-1_Diff10'
DATASET_ClearedDataset2_1_Diff20 = 'ClearedDataset2-1_Diff20'
DATASET_P234WithLittleP1Train = 'P234WithLittleP1Train'
DATASET_P134WithLittleP2Train = 'P134WithLittleP2Train'
DATASET_P124WithLittleP3Train = 'P124WithLittleP3Train'
DATASET_PXWithLittlePXTest = 'PXWithLittlePXTest'
DATASET_P12WithLittleP3Train = 'P12WithLittleP3Train'
DATASET_P13WithLittleP2Train = 'P13WithLittleP2Train'
DATASET_P23WithLittleP1Train = 'P23WithLittleP1Train'
DATASET_P123WithLittleTest = 'P123WithLittleTest'
DATASET_P12WithLittleP3Train_Sample100 = 'P12WithLittleP3Train_Sample100'
DATASET_P13WithLittleP2Train_Sample100 = 'P13WithLittleP2Train_Sample100'
DATASET_P23WithLittleP1Train_Sample100 = 'P23WithLittleP1Train_Sample100'
DATASET_P123WithLittleTest_Sample100 = 'P123WithLittleTest_Sample100'

DATASET_Cleared1_ModelB_S1_10 = 'Cleared1_ModelB_S1-10'
DATASET_ClearedDataset_1_ModelB_S1_5 = 'ClearedDataset-1_ModelB_S1-5'
DATASET_Cleared1_ModelB_S11_15 = 'Cleared1_ModelB_S11-15'
DATASET_Cleared1_ModelBCD_S11_40 = 'Cleared1_ModelBCD_S11-40'
OUTPUT_DIR = '/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF'

# Variable
common_config_model = 'resnet18'
common_config_train_set = DATASET_Cleared1_ModelB_S1_10
common_config_test_set = common_config_train_set
common_config_devices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# mix training
# common_config_train_point = ['1', '2']
# common_config_test_point = ['1', '2']
# common_config_train_snr = ['5', '10', '15', '20', '25', '30']
# common_config_test_snr = ['25']

# cross training
common_config_train_point = ['1', '2']
common_config_test_point = ['1', '2']
common_config_train_snr = ['no']  # ['15', '20', '25', 'no']
common_config_test_snr = ['no']  # ['15', '20', '25', 'no']
common_config_device_count = len(common_config_devices)
common_config_learning_rate = 0.0001
common_config_batch_size = 128
common_config_diff_channel = []
