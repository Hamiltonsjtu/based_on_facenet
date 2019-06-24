import requests
import json
import os
# #######-------------image client------------#######
img_path = "F:/xi.jpg"

files = {"file": open(img_path, "rb")}
r = requests.post("http://0.0.0.0:5000/upload", files=files)
returnval = json.loads(r.text)


people_en_ch = {'xijinping': '习近平', 'hujintao': '胡锦涛', 'jiangzemin': '江泽民', 'dengxiaoping': '邓小平', 'wenjiabao': '温家宝', 'maozedong': '毛泽东', 'zhouenlai': '周恩来'}

if returnval['code'] == 301:
    print('图片: {} 是{}的, 图片中有{}张政治敏感人物'.format(returnval['file'], returnval['result'], len(returnval['data'])))
    data_min = returnval['data']
    print('图中政治敏感人物为: ')
    for i in range(len(data_min)):
        print('                 ' + people_en_ch[data_min[i]['user_name']] + '相似度为: ' + '{:.2%}'.format(data_min[i]['score']))
elif returnval['code'] == 300:
    print('图片: {} 有人脸是{}的'.format(returnval['file'], returnval['result']))
elif returnval['code'] == 200:
    print('图片: {} 没有人脸是{}的'.format(returnval['file'], returnval['result']))

