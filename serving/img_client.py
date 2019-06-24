import requests
import json
import os
import numpy as np
import cv2

from scipy import misc
from detect_face_np import load_and_align_data
from process import faceNet_serving_V0

# #######-------------image client------------#######

peoples = ['jiangzemin', 'hujintao', 'xijinping', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
# peoples = ['xijinping']

result = {}


def img_restore(img_path, returnval, people):
    img = cv2.imread(img_path, 3)
    img_head, img_name = os.path.split(img_path)
    file_name, file_extension = os.path.splitext(img_name)
    if returnval['message'] == 'Has_no_faces':

        font = cv2.FONT_HERSHEY_SIMPLEX
        img_add = cv2.putText(img, 'No_Face!', (10, 10), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        dir = 'E:/test_Re_1/' + people + '/' + 'No_face/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + img_name, img_add)

    elif returnval['message'] == 'Has_face_pass':
        det_arr_ = returnval['det_arr'][0]
        num_face = len(det_arr_)//4
        det_arr = np.reshape(det_arr_, (num_face, 4))
        for i in range(num_face):
            det_arr_slice = det_arr[i,:]
            bb = np.array(det_arr_slice, dtype=np.int32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_rec = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0))
            img_add = cv2.putText(img_rec, 'has Face!', (10, 10), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            dir = 'E:/test_Re_1/' + people + '/' + 'detect_face_is_passed/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            cv2.imwrite(dir + file_name + '_' + str(i) + file_extension, img_add)

    else:
        det_arr_ = returnval['det_arr'][0]
        num_face = len(det_arr_)//4
        det_arr = np.reshape(det_arr_, (num_face, 4))
        data = returnval['data']
        for i in range(np.shape(det_arr)[0]):
            det_arr_slice = det_arr[i,:]
            bb = np.array(det_arr_slice, dtype=np.int32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_rec = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0))
            img_add = cv2.putText(img_rec, data[i]['user_name'], (bb[0], bb[3]), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            dir = 'E:/test_Re_1/' + people + '/' + 'not_passed/' + data[i]['user_name'] + '/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            cv2.imwrite(dir + file_name + '_' + str(i) + file_extension, img_add)


for name in peoples:

    image_dir = 'F:/peoples_baidu/' + name + '_baidu'

    image_pic = os.listdir(image_dir)
    num_all = 0
    num_care = 0
    num_right_people = 0
    for i in image_pic:
        img_path = os.path.join(image_dir, i)
        # img_path = 'E:/shuai/Face/TZX/image_test/xi_hu2.jpg'
        # img_path = 'F:/peoples_baidu/jiangzemin_baidu/146.jpg'
        # img = misc.imread(os.path.expanduser(img_path), mode='RGB')
        # faces, det_arr = load_and_align_data(img)

        files = {"file": open(img_path, "rb")}
        # r = requests.post("http://192.168.1.254:5001/v1/face_censor", files=files)
        r = requests.post("http://0.0.0.0:5000/upload", files=files)
        returnval = json.loads(r.text)

        print(returnval)

        # image = cv2.imread(img_path)
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('f:/gray.jpg', image_gray)

        img_restore(img_path, returnval, name)

        if returnval['result'] == '不合规':
            num_care += 1
            face_num = len(returnval['data'])
            for j in range(face_num):
                if returnval['data'][j]['user_name'] == name:
                    num_right_people += 1
        num_all += 1

    result[name] = {'num_care': num_care, 'num_all': num_all, 'precision': num_care/num_all, 'num_right_one':num_right_people, 'accuracy': num_right_people/num_all}
    print('num_care: {}, all {} and precision {}'.format(num_care, num_all, num_care/num_all))
    print('num of detect the right one is {}, and accuracy is {}'.format(num_right_people, num_right_people/num_all))


np.save('result_without_aligned_V1.npy', result)


