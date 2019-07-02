import requests
import json
import os
import numpy as np
import cv2

from scipy import misc
from detect_face_np import load_and_align_data
from process import faceNet_serving_V0

# #######-------------image client------------#######

peoples = ['xijinping', 'jiangzemin', 'hujintao', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
# peoples = ['xijinping']
RESULT_NUM_att = ['Not_detect', 'Detect_Pass', 'Not_Pass']

result = {}
for i in peoples:
    result[i] = {}
    for j in RESULT_NUM_att[0:-1]:
        result[i][j] = 0
    data_sub = {}
    for k in peoples:
        data_sub[k] = []
    result[i][RESULT_NUM_att[-1]] = data_sub


def img_restore(img_path, returnval, people):
    img = cv2.imread(img_path, 3)
    img_head, img_name = os.path.split(img_path)
    file_name, file_extension = os.path.splitext(img_name)
    if returnval['message'] == 'Has_no_faces':
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_add = cv2.putText(img, 'No_Face!', (10, 10), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        dir = 'E:/test_Re_diffthred/' + people + '/' + 'No_face/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + img_name, img_add)
        result[people]['Not_detect'] += 1
    elif returnval['message'] == 'Has_face_pass':
        det_arr_ = returnval['det_arr'][0]
        num_face = len(det_arr_)//4
        det_arr = np.reshape(det_arr_, (num_face, 4))
        for i in range(num_face):
            det_arr_slice = det_arr[i,:]
            bb = np.array(det_arr_slice, dtype=np.int32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_rec = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0))
            img_add = cv2.putText(img_rec,  returnval['user_name'][:2] + returnval['score'][:3], (bb[0], bb[3]), font, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
            dir = 'E:/test_Re_diffthred/' + people + '/' + 'detect_face_is_passed/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            cv2.imwrite(dir + file_name + '_' + str(i) + file_extension, img_add)
            result[people]['Detect_Pass'] += 1
    else:
        # det_arr_ = returnval['det_arr'][0]
        # num_face = len(det_arr_)//4
        # det_arr = np.reshape(det_arr_, (num_face, 4))
        # data = returnval['data']
        for i in range(len(returnval['data'])):
            det_arr_slice = returnval['data'][i]['det_arr']
            bb = np.squeeze(np.array(det_arr_slice, dtype=np.int32))
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_rec = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0))
            img_add = cv2.putText(img_rec, returnval['data'][i]['user_name'][:2] + str(returnval['data'][i]['score'])[:3], (bb[0], bb[3]), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            dir = 'E:/test_Re_diffthred/' + people + '/' + 'not_passed/' + returnval['data'][i]['user_name'] + '/'
            if not os.path.exists(dir):
                 os.makedirs(dir)
            cv2.imwrite(dir + file_name + '_' + str(i) + file_extension, img_add)
            result[people]['Not_Pass'][returnval['data'][i]['user_name']].append(returnval['data'][i]['score'])

    np.save('Accuracy_Precision' + 'diffthred' + '.npy', result)

def main():
    #
    # for name in peoples:
    #     image_dir = 'F:/peoples_baidu/' + name + '_baidu'
    #     image_pic = os.listdir(image_dir)
    #     for i in image_pic:
    #         img_path = os.path.join(image_dir, i)
    #         files = {"file": open(img_path, "rb")}
    #         # r = requests.post("http://192.168.1.254:5001/v1/face_censor", files=files)
    #         r = requests.post("http://0.0.0.0:5000/upload", files=files)
    #         returnval = json.loads(r.text)
    #         print(returnval)
    #         img_restore(img_path, returnval, name)

    image_dir = 'F:/test/test_xijinping'
    image_pic = os.listdir(image_dir)
    for i in image_pic:
        img_path = os.path.join(image_dir, i)
        files = {"file": open(img_path, "rb")}
        # r = requests.post("http://192.168.1.254:5001/v1/face_censor", files=files)
        r = requests.post("http://0.0.0.0:5000/upload", files=files)
        returnval = json.loads(r.text)
        print(returnval)
        name = 'xijinping'
        img_restore(img_path, returnval, name)


if __name__ == '__main__':
    main()


