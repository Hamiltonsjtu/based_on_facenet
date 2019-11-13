import numpy as np
import cv2
import os
import pandas as pd
img_dir = r'F:\FocusScreen\FocusImage'

anno = pd.read_csv([img_dir + '/' + i for i in os.listdir(img_dir) if i.startswith('eyepo')][0], sep = '\t')
anno_unique = anno.drop_duplicates(subset=['name'], keep='first').reset_index()         # 所有图片
anno_dropD = anno.drop_duplicates(subset=['name'], keep=False).reset_index()            # 去掉重复的
anno_Dupli = anno_unique.append(anno_dropD).drop_duplicates(keep=False).reset_index()   # 重复的
anno_allDupli = anno[anno['name'].isin(anno_Dupli['name'])].reset_index()               # 保留重复的

base_filename = 'Duplicated_img.txt'
anno_Dupli.to_csv(os.path.join(img_dir, base_filename))

base_filename = 'Duplicated_all_img.txt'
anno_allDupli.to_csv(os.path.join(img_dir, base_filename))

base_filename = 'not_Duplicated_img.txt'
anno_dropD.to_csv(os.path.join(img_dir, base_filename))

