from shutil import copy


if __name__ == '__main__':
    source_dir = 'D:/WIFI_Dataset/AugData/ClearedDataset-20/P2/P2-'
    target_dir = 'D:/WIFI_Dataset/AugData/LittleRawAug/P2/'
    mylist = ['', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in mylist:
        source_file = source_dir + i + '1.mat'
        copy(source_file, target_dir)
        source_file = source_dir + i + '6.mat'
        copy(source_file, target_dir)
