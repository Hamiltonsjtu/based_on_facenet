import requests
import json
import os
import numpy as np
import cv2
from scipy import misc

# #######-------------image client------------#######

peoples = ['xijinping', 'jiangzemin', 'hujintao', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']

def main():

    for name in peoples:
        image_dir = 'F:/peoples_baidu_test/' + name
        image_pic = os.listdir(image_dir)
        for i in image_pic:
            img_path = os.path.join(image_dir, i)
            files = {"file": open(img_path, "rb")}
            r = requests.post("http://192.168.1.37:5000/v1/face_censor", files=files)
            #r = requests.post("http://192.168.1.254:5001/v1/face_censor", files=files)
            # r = requests.post("http://0.0.0.0:5000/upload", files=files)
            returnval = json.loads(r.text)
            print(returnval)


if __name__ == '__main__':
    main()


