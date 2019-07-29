import requests
import json
import os
import numpy as np
import cv2
from scipy import misc


source_dir = 'F:/peoples_google'
# source_dir = 'F:/TEST'

# source_dir = 'F:/raw_image_web_crop'
peoples = os.listdir(source_dir)

def main():
    for name in peoples:
        # image_dir = 'F:/peoples_baidu/' + name + '_baidu'
        image_dir = source_dir + '/' + name
        image_pic = os.listdir(image_dir)
        for i in image_pic:
            img_path = os.path.join(image_dir, i)
            files = {"file": open(img_path, "rb")}
            r = requests.post("http://192.168.1.23:5003/ssd_face", files=files)
            # r = requests.post("http://192.168.1.254:5002/v2/face_censor", files=files)
            # r = requests.post("http://0.0.0.0:5000/v2/face_censor", files=files)
            # r = requests.post("http://0.0.0.0:5000/upload", files=files)
            returnval = json.loads(r.text)

            if len(returnval['bbox']) > 0:
                save_cropped_faces(img_path, returnval['bbox'], img_path)
            print(returnval)


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def save_cropped_faces(img_path, bb_list, image_pic, image_size=160):
    num_face = np.size(bb_list) // 4
    bb_box = np.reshape(bb_list, (num_face, 4))
    try:
        img = misc.imread(img_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(img_path, e)
        print(errorMessage)
    else:
        if img.ndim == 2:
            img = to_rgb(img)
    img = img[:, :, 0:3]
    dst_dir, _filename = os.path.split(image_pic)
    file_name, file_extension = os.path.splitext(_filename)
    output_dir = os.path.split(dst_dir)[0] + '_crop' + '/' + os.path.split(dst_dir)[1] + '_cropped'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_face):
        bb = bb_box[i, :]
        # img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0))
        # img = cv2.putText(img, str(i), (bb[0], bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        output_filename_n = output_dir + '/' + file_name + '_' + str(i) + file_extension
        print('image path {}, shape {} and cropped size {}'.format(img_path, np.shape(img), bb))
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if np.shape(cropped)[2] != 3:
            print('#{} channel is not suitable for misc save!'.format(img_path))
            continue
        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        misc.imsave(output_filename_n, scaled)


if __name__ == '__main__':
    main()


