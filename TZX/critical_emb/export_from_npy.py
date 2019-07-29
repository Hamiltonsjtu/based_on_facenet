import numpy as np

emb = np.load('embeddings.npy')
labels_str = np.load('label_strings.npy')
labels_num = np.load('labels.npy')

peoples = list(set(labels_str))

emb_dict = {}

for i in peoples:
    index = np.where(labels_str == i)[0]
    emb_ = emb[index, :]
    emb_ave = np.mean(emb_, axis = 0)
    emb_dict[i] = {'emb': emb_, 'emb_ave': emb_ave}

np.save('aligned_embedings_dict.npy', emb_dict)
