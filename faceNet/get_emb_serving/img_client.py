import requests
import json
import os
import numpy as np
import cv2
import pandas as pd

# #######-------------image client------------#######

peoples = ['jiangzemin', 'hujintao', 'xijinping', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']

result = {}


class ImageEmbClass():
    "Stores the paths to images for a given class"

    def __init__(self, class_name, name, emb_vector):
        self.cls = class_name
        self.name = name
        self.emb = emb_vector

    def __str__(self):
        return self.cls + ',' + self.name + ', ' + str(len(self.emb))

    def __len__(self):
        return len(self.cls)


dataset = []


image_dir = 'F:/baidu_crop/'

# for i in image_pic:
    # img_path = os.path.join(image_dir, i)

    # img_path = 'F:/baidu_crop/dengxiaoping_baidu/0.png'
    # img_path = 'F:/peoples_baidu/jiangzemin_baidu/146.jpg'
    # img = misc.imread(os.path.expanduser(img_path), mode='RGB')
    # faces, det_arr = load_and_align_data(img)

path_exp = os.path.expanduser(image_dir)
print('path expanduser is {}'.format(path_exp))
classes = [path for path in os.listdir(path_exp) \
           if os.path.isdir(os.path.join(path_exp, path))]
classes.sort()
nrof_classes = len(classes)
print('have {} class(es), and class is {}'.format(nrof_classes, classes))
for i in range(nrof_classes):
    class_name = classes[i]
    facedir = os.path.join(path_exp, class_name)
    for j in os.listdir(facedir):
        img_path = os.path.join(facedir,j)
        file_name = j

        files = {"file": open(img_path, "rb")}
        # r = requests.post("http://192.168.1.254:5001/v1/face_censor", files=files)
        r = requests.post("http://0.0.0.0:5000/upload", files=files)
        returnval = json.loads(r.text)

        dataset.append(ImageEmbClass(class_name, file_name, returnval['emb']))

        print(returnval)


        # image_show = cv2.cvtColor(np.reshape(int(returnval['image_data']), (182,182, 3)))
        # cv2.imshow(image_show)
        # cv2.waitKey()



