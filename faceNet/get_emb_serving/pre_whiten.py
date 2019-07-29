import numpy as np
from scipy import misc
import os
import cv2

def load_data(image_paths, image_size):

    images = np.zeros((image_size, image_size, 3))
    img = misc.imread(image_paths)
    # print('pre-wihten', img)
    if img.ndim == 2:
        img = to_rgb(img)

    img = prewhiten(img)
    images[:,:,:] = img
    images = misc.imresize(images,(160, 160))
    # print('after-wihter', images)
    return images


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def main():
    # train_data_dir = 'F:/baidu_crop_06271503'
    train_data_dir = 'F:/raw_image_web_crop_multi'

    folders = os.listdir(train_data_dir)
    for i in folders:
        for j in os.listdir(os.path.join(train_data_dir, i)):
            img_path = train_data_dir + '/' + i + '/' + j
            images = load_data(img_path, 182)
            dst_img_dir = train_data_dir + '_160/' + i
            if not os.path.exists(dst_img_dir):
                os.makedirs(dst_img_dir)
            dst_img_path = dst_img_dir + '/' + j

            misc.imsave(dst_img_path, images)

if __name__ == '__main__':
    main()