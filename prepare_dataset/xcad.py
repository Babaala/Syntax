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



def get_path_list(root_path,img_path,label_path,fov_path):
    tmp_list = [img_path,label_path,fov_path]
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
        f.write(str(name_list[0][i]) + " " + str(name_list[1][i]) + " " + str(name_list[2][i]) + '\n')
    f.close()


if __name__ == "__main__":
    # ------------Path of the dataset -------------------------
    data_root_path = r'G:\Datasets\XCAD'
    # if not os.path.exists(data_root_path): raise ValueError("data path is not exist, Please make sure your data path is correct")
    # train
    img_train = "test/images/"
    gt_train = "test/masks/"
    fov_train = "test/fov/"

    # test
    img_test = "test/val/"
    gt_test = "test/masks_val/"
    fov_test = "test/fov_val/"

    # ----------------------------------------------------------
    save_path = "./data_path_list/XCAD/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    train_list = get_path_list(data_root_path,img_train,gt_train,fov_train)
    print('Number of train imgs:', len(train_list[0]))
    write_path_list(train_list, save_path, 'train.txt')

    test_list = get_path_list(data_root_path,img_test,gt_test,fov_test)
    print('Number of test imgs:', len(test_list[0]))
    print('Number of test imgs:', len(test_list[0]))
    write_path_list(test_list, save_path, 'test.txt')

    print("Finish!")

