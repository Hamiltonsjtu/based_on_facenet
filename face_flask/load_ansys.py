import numpy as np

# accuracy_precision = np.load('./Accuracy_Precision.npy').item()
emb_old = np.load('./people_embs.npy')
emb = np.load('./20190627_emb_feature_large_2_160/embeddings.npy')
labels_str = np.load('./20190627_emb_feature_large_2_160/label_strings.npy')
labels_num = np.load('./20190627_emb_feature_large_2_160/labels.npy')
# DATA = {}
# DATA['emb'] = emb
# DATA['labels_str'] = labels_str
# DATA['labels_num'] = labels_num
emb_ave = {}

peoples = list(set(labels_str))
for i in peoples:
    index = list(np.where(labels_str == i)[0])
    emb_ave[i] = np.mean(emb[index, :], axis= 0)

key = [i for i,_ in emb_ave.items()]
value = [i for _, i in emb_ave.items()]

old_05 = np.load('./Accuracy_Precision0.5.npy').item()
small_05 = np.load('./Accuracy_Precision0.5_small.npy').item()

for i, name in old_05.items():
    for j, value in old_05[i]['Not_Pass'].items():
        old_05[i]['Not_Pass'][j] = np.array(value)

for i, name in small_05.items():
    for j, value in small_05[i]['Not_Pass'].items():
        small_05[i]['Not_Pass'][j] = np.array(value)


print('OK')


