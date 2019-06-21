import requests
import json
import os
import numpy as np
# #######-------------image client------------#######

peoples = ['jiangzemin', 'hujintao', 'xijinping', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
result = {}


for name in peoples:

    image_dir = 'F:/peoples_baidu/' + name + '_baidu'

    image_pic = os.listdir(image_dir)
    num_all = 0
    num_care = 0
    num_right_people = 0
    for i in image_pic:
        img_path = os.path.join(image_dir, i)
        files = {"file": open(img_path, "rb")}
        # r = requests.post("http://192.168.1.254:5001/v1/face_censor", files=files)
        r = requests.post("http://0.0.0.0:5000/upload", files=files)
        returnval = json.loads(r.text)
        print(returnval)

        if returnval['result'] == '不合规':
            num_care += 1
            face_num = len(returnval['data'])
            for j in range(face_num):
                if returnval['data'][j]['user_name'] == name:
                    # print('!!!!FIND THE RIGHT ONE!!!!!')
                    num_right_people += 1

        num_all += 1
    result[name] = {'num_care': num_care, 'num_all': num_all, 'precision': num_care/num_all, 'num_right_one':num_right_people, 'accuracy': num_right_people/num_all}
    print('num_care: {}, all {} and precision {}'.format(num_care, num_all, num_care/num_all))
    print('num of detect the right one is {}, and accuracy is {}'.format(num_right_people, num_right_people/num_all))


np.save('result_without_aligned_V1.npy', result)

# people_en_ch = {'xijinping': '习近平', 'hujintao': '胡锦涛', 'jiangzemin': '江泽民', 'dengxiaoping': '邓小平', 'wenjiabao': '温家宝', 'maozedong': '毛泽东', 'zhouenlai': '周恩来'}
#
# if returnval['code'] == 301:
#     print('图片: {} 是{}的, 图片中有{}张政治敏感人物'.format(returnval['file'], returnval['result'], len(returnval['data'])))
#     data_min = returnval['data']
#     print('图中政治敏感人物为: ')
#     for i in range(len(data_min)):
#         print('                 ' + people_en_ch[data_min[i]['user_name']] + '相似度为: ' + '{:.2%}'.format(data_min[i]['score']))
# elif returnval['code'] == 300:
#     print('图片: {} 有人脸是{}的'.format(returnval['file'], returnval['result']))
# elif returnval['code'] == 200:
#     print('图片: {} 没有人脸是{}的'.format(returnval['file'], returnval['result']))

