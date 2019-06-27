import numpy as np
# class_names = np.load('class.npy')
# file_name = np.load('name.npy')
# embs = np.load('embs.npy')
#
# cls_names = list(set(class_names))
# data = {}
# data_ave = {}
# for i in cls_names:
#     indice = np.where(class_names == i)[0]
#     data[i] = embs[indice,:]
#     data_ave[i] = np.mean(data[i], axis=0)
#
# for i in data_ave:
#     print(i)
#     # print(data_ave[i])


peoples = ['xijinping', 'jiangzemin', 'hujintao', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
# peoples = ['xijinping']
RESULT_NUM_att = ['Not_detect', 'Detect_Pass', 'Not_Pass']


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