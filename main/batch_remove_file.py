from shutil import copy
import os


if __name__ == '__main__':
    source_dir = 'D:/WIFI_Dataset/AugData/P123WithLittleTest/P2/P2-'
    mylist = ['', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in mylist:
        source_file = source_dir + i + '1.mat'
        os.remove(source_file)
        source_file = source_dir + i + '6.mat'
        os.remove(source_file)
