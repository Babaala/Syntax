# =========================================================
#
#
#
# =========================================================
import os
import h5py, cv2, imageio
import numpy as np
from PIL import Image
from os.path import join


def get_path_list(root_path,img_path):
    tmp_list = [img_path]
    res = []
    for i in range(len(tmp_list)):
        data_path = join(data_root_path,tmp_list[i])
        filename_list = os.listdir(data_path)
        filename_list.sort()
        res.append([join(data_path,j) for j in filename_list])
    return res


def write_path_list(name_list, save_path, file_name):
    f = open(join(save_path, file_name), 'w')
    for i in range(len(name_list[0])):
        f.write(str(name_list[0][i]) + " E:/Datasets/XiangMu/masks/100.png" + " E:/Datasets/XiangMu/fov/100.png" + '\n')
    f.close()


if __name__ == "__main__":
    # ------------Path of the dataset -------------------------
    data_root_path = r'E:\Datasets\XiangMu'
    # if not os.path.exists(data_root_path): raise ValueError("data path is not exist, Please make sure your data path is correct")

    # test
    img_test = "new/images/"

    # ----------------------------------------------------------
    save_path = "./data_path_list/XiangMu/new/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    test_list = get_path_list(data_root_path,img_test)
    print('Number of test imgs:', len(test_list[0]))
    write_path_list(test_list, save_path, 'test.txt')

    print("Finish!")