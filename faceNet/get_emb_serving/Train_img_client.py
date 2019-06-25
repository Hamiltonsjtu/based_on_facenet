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


class_names = []
file_names = []
embs = []

image_dir = 'F:/baidu_crop/'

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

        class_names.append(class_name)
        file_names.append(file_name)
        embs.append(returnval['emb'])
        print(returnval)

np.save('class.npy', class_names)
np.save('name.npy', file_names)
np.save('embs.npy', embs)


