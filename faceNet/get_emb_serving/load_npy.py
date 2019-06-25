import numpy as np
class_names = np.load('class.npy')
file_name = np.load('name.npy')
embs = np.load('embs.npy')

cls_names = list(set(class_names))
data = {}
data_ave = {}
for i in cls_names:
    indice = np.where(class_names == i)[0]
    data[i] = embs[indice,:]
    data_ave[i] = np.mean(data[i], axis=0)

print(class_names.shape)