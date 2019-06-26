from scipy import misc
import cv2


wihten_img_src = 'F:/baidu_crop_wihten/dengxiaoping_baidu/0.png'
img_src = 'F:/baidu_crop/dengxiaoping_baidu/0.png'


wihten_img_misc = misc.imread(wihten_img_src)
wihten_img_cv = cv2.imread(wihten_img_src)

img_misc = misc.imread(img_src)
img_cv = cv2.imread(img_src)

print('test!')