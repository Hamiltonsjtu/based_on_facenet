import numpy as np
import matplotlib.pyplot as plt

theta = {'xijinping': [0.76],
         'hujintao': [0.76],
         'jiangzemin': [0.76],
         'dengxiaoping': [0.75],
         'wenjiabao': [0.75],
         'maozedong': [0.83],
         'zhouenlai': [0.83]}

a = {'zhu': 1}

print(type(str(a.keys())))
print(str(a.keys()))

diff_thre = list([i for _, i in theta.items()])
diff_name = list([i for i, _ in theta.items()])

print('max likely its location {} and its name {}'.format(np.argmax(diff_thre), theta[diff_name[np.argmax(diff_thre)]]))

thre_min = np.min(diff_thre)


accuracy_precision = np.load('./Accuracy_Precision0.5.npy').item()


for i, name in accuracy_precision.items():
    plt.figure()
    j_num = 1
    for j, value in accuracy_precision[i]['Not_Pass'].items():
        accuracy_precision[i]['Not_Pass'][j] = np.array(value)
        plt.subplot(2,4,j_num)
        plt.hist(accuracy_precision[i]['Not_Pass'][j])
        j_num += 1
        plt.title(i + ' / ' + j)
    plt.show()


for i, _ in accuracy_precision.items():
    plt.figure()
    j_num = 1
    for j, value in accuracy_precision.items():
        plt.subplot(2,4,j_num)
        plt.hist(accuracy_precision[j]['Not_Pass'][i])
        j_num += 1
        plt.title(j + ' / ' + i)
    plt.show()

class_names = np.load('./20190627_emb_feature_1/label_strings.npy')
file_name = np.load('./20190627_emb_feature_1/labels.npy')
embs = np.load('./20190627_emb_feature_1/embeddings.npy')

cls_names = list(set(class_names))
data = {}
data_ave = {}
for i in cls_names:
    indice = np.where(class_names == i)[0]
    data[i] = embs[indice,:]
    data_ave[i] = np.mean(data[i], axis=0)

for i in data_ave:
    print(i)
    # print(data_ave[i])




# def multi(*args):
#     """
#     Build multiple level dictionary for python
#     For example:
#         multi(['a', 'b'], ['A', 'B'], ['1', '2'], {})
#     returns
#         {   'a': {'A': {'1': {}, '2': {}}, 'B': {'1': {}, '2': {}}},
#             'b': {'A': {'1': {}, '2': {}}, 'B': {'1': {}, '2': {}}}}
#     """
#     if len(args) > 1:
#         return {arg: multi(*args[1:]) for arg in args[0]}
#     else:
#         return args[0]
#
#
# result = multi(peoples, RESULT_NUM_att, {})

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
        data_sub[k] = 0
    result[i][RESULT_NUM_att[-1]] = data_sub
# not_detct_num = 0


# print(class_names.shape)