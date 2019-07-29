import requests
import json
import os
import numpy as np
import cv2
import pandas as pd

# #######-------------image client------------#######
image_dir = r'F:\SSD_cropped'
# image_dir = r'F:\TEST'
peoples = os.listdir(image_dir)

result = {}

class_names = []
file_names = []
embs = []

# image_dir = 'F:/baidu_crop_wihten_160/'

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
    print('# {} class, name {} '.format(i, class_name))
    for j in os.listdir(facedir):
        img_path = os.path.join(facedir,j)
        file_name = j
        files = {"file": open(img_path, "rb")}
        r = requests.post("http://192.168.1.23:5006/upload", files=files)
        print('request STATUS ', r.status_code)
        returnval = json.loads(r.text)
        class_names.append(class_name)
        file_names.append(file_name)
        embs.append(returnval['emb'])
        print(returnval)

    print('{} class exported ! '.format(class_name))

np.save('class.npy', class_names)
np.save('name.npy', file_names)
np.save('embs.npy', embs)
