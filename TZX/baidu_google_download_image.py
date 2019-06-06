#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import os
# import pandas as pd
# import numpy as np
import re
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time


# In[18]:


#百度
data = {}
data['xijinping'] = 'http://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E4%B9%A0%E8%BF%91%E5%B9%B3&step_word=&hs=0&pn=1&spn=0&di=56350&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=3195191670%2C4243836286&os=1750335976%2C2169695471&simid=999382772%2C900902286&adpicid=0&lpn=0&ln=705&fr=&fmq=1559270954167_R&fm=detail&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=11&oriquery=&objurl=http%3A%2F%2Fnews.cnr.cn%2Fnative%2Fgd%2F20190526%2FW020190526362802482996.png&fromurl=ippr_z2C%24qAzdH3FAzdH3Fgjof_z%26e3Bvg6_z%26e3BvgAzdH3FgwptejAzdH3F21AzdH3Fda8lacdmAzdH3Fpda8lacdm_cd9mdmmnm_z%26e3Bfip4s%3Fu654%3Dftg2sj4jffw2j%26tfwrrtgfpwssj1%3Da&gsm=0&rpstart=0&rpnum=0&islist=&querylist=&force=undefined'
data['hujintao'] = 'http://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E8%83%A1%E9%94%A6%E6%B6%9B&step_word=&hs=0&pn=0&spn=0&di=17160&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=3569089730%2C4012485771&os=1668970396%2C5659709&simid=4128752337%2C372500585&adpicid=0&lpn=0&ln=1520&fr=&fmq=1559271955544_R&fm=detail&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=http%3A%2F%2Fphotocdn.sohu.com%2F20080316%2FImg255729141.jpg&fromurl=ippr_z2C%24qAzdH3FAzdH3Fgjof_z%26e3Bf5i7_z%26e3Bv54AzdH3Fdaaban8mAzdH3Fgdcc0dl89a_z%26e3Bfip4s&gsm=0&rpstart=0&rpnum=0&islist=&querylist=&force=undefined'
data['jiangzemin'] = 'http://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E6%B1%9F%E6%B3%BD%E6%B0%91&step_word=&hs=0&pn=0&spn=0&di=88440&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=3931291751%2C3655415633&os=2408426679%2C964684936&simid=4081438931%2C638315372&adpicid=0&lpn=0&ln=1365&fr=&fmq=1559271998992_R&fm=detail&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=http%3A%2F%2Fwww.li-xian.gov.cn%2Fzj%2FUploadFiles_4992%2F201308%2F2013080509380564.jpg&fromurl=ippr_z2C%24qAzdH3FAzdH3Fooo_z%26e3Bst-xtwg_z%26e3B25e_z%26e3BvgAzdH3Fz3AzdH3FSi5oA6ptvsj_z%26e3Bwfr%3FA6ptvsjID%3D80a9m&gsm=0&rpstart=0&rpnum=0&islist=&querylist=&force=undefined'
data['wenjiabao'] = 'http://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E6%B8%A9%E5%AE%B6%E5%AE%9D&step_word=&hs=0&pn=0&spn=0&di=186010&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=2069832667%2C318234332&os=709286399%2C122064579&simid=3512413508%2C382995928&adpicid=0&lpn=0&ln=1520&fr=&fmq=1559272040584_R&fm=detail&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=http%3A%2F%2Fwww.gov.cn%2Fgovweb%2Fjrzg%2Fimages%2Fimages%2F1c6f6506c5d50ff6771301.jpg&fromurl=ippr_z2C%24qAzdH3FAzdH3Fooo_z%26e3B25e_z%26e3BvgAzdH3F25eojkAzdH3F36z2AzdH3Fda88-8aAzdH3FacAzdH3Fv5gpjgp_8lmn88d_z%26e3Bip4&gsm=0&rpstart=0&rpnum=0&islist=&querylist=&force=undefined'
data['maozedong'] = 'http://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E6%AF%9B%E6%B3%BD%E4%B8%9C&step_word=&hs=0&pn=0&spn=0&di=131890&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=85565646%2C3355969318&os=2806044701%2C697785234&simid=3522999130%2C560162033&adpicid=0&lpn=0&ln=1560&fr=&fmq=1559272075106_R&fm=detail&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=http%3A%2F%2Fimg1.cache.netease.com%2Fcatchpic%2FF%2FF8%2FF89A480FFF9BB44C19F753F809045CAD.jpg&fromurl=ippr_z2C%24qAzdH3FAzdH3Fgjof_z%26e3B8mn_z%26e3Bv54AzdH3F88AzdH3FalalAzdH3FalAzdH3F0DGHVCHlaaa89JBc_d_z%26e3Bip4s&gsm=0&rpstart=0&rpnum=0&islist=&querylist=&force=undefined'
data['zhouenlai'] = 'http://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E5%91%A8%E6%81%A9%E6%9D%A5&step_word=&hs=0&pn=0&spn=0&di=150370&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=2030321234%2C1737131565&os=1228163468%2C258681343&simid=4286497428%2C781952079&adpicid=0&lpn=0&ln=1304&fr=&fmq=1559272117671_R&fm=detail&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=http%3A%2F%2Fs9.rr.itc.cn%2Fr%2FwapChange%2F20149_10_10%2Fa6d2hq5486748678323.jpg&fromurl=ippr_z2C%24qAzdH3FAzdH3F4_z%26e3Bf5i7_z%26e3Bv54AzdH3FgAzdH3F9a98lmn0aAzdH3F&gsm=0&rpstart=0&rpnum=0&islist=&querylist=&force=undefined'
data['dengxiaoping'] = 'http://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E9%82%93%E5%B0%8F%E5%B9%B3&step_word=&hs=0&pn=0&spn=0&di=106590&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=1809598407%2C434214029&os=1957159159%2C4077701768&simid=4259159085%2C789364792&adpicid=0&lpn=0&ln=1608&fr=&fmq=1559272140043_R&fm=detail&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=http%3A%2F%2Fimg1.cache.netease.com%2Fcatchpic%2FA%2FAA%2FAA40189605FE0160AD59474E4F349158.jpg&fromurl=ippr_z2C%24qAzdH3FAzdH3Fgjof_z%26e3B8mn_z%26e3Bv54AzdH3F8cAzdH3Fam8lAzdH3F8mAzdH3FASGnImGAaaa89JBm_9_z%26e3Bip4s&gsm=0&rpstart=0&rpnum=0&islist=&querylist=&force=undefined'

# peoples = ['hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
peoples = ['xijinping']
browser = webdriver.Chrome()

for people in peoples:
    print(people)
    browser.get(data[people])

    src_path = os.path.join(os.getcwd(), people+'_baidu')
    if not os.path.exists(src_path):
        os.mkdir(src_path)

    address = browser.find_element_by_id("currentImg").get_attribute(name='src')
    old = address
    num = 0
    for i in range(1000):
        try:
            if num%50 == 0:
                print('current num {}'.format(num))
            #wait=WebDriverWait(browser,10)
            #wait.until(EC.presence_of_element_located((By.ID,'currentImg')))
            browser.find_element_by_class_name("img-next").click()
            time.sleep(3)
            address=browser.find_element_by_id("currentImg").get_attribute(name='src')

            if old == address:
                continue
            else:
                old= address
                content = requests.get(address)

                if content.status_code == 200:

                    with open(src_path + '/' + '%d.jpg'%num, 'wb') as pic_fd:
                        num += 1
                        pic_fd.write(content.content)
                else:
                    continue
        except:
            break


# #谷歌1
# zombie_list=[]
# zombie='https://www.google.com/search?q=%E6%81%90%E6%80%96%E8%A1%80%E8%85%A5&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjI4r788YfgAhXNc94KHSXXBfoQ_AUIDigB&biw=1920&bih=938#imgrc=2kRZQPzh-l2EuM:'
# browser = webdriver.Chrome()
# browser.get(zombie)
#
# sub_list=re.findall('img class="irc_mi" src="(.*?)"', browser.page_source)
# zombie_list.extend(sub_list)
#
# for i in range(2000):
#     try:
#         if i % 50 == 0:
#             print(i)
#         browser.find_element_by_id("irc-rac").click()
#         time.sleep(3)
#         sub_list = re.findall('img class="irc_mi" src="(.*?)"',browser.page_source)
#         if len(sub_list) > 0:
#             zombie_list.extend(sub_list)
#         else:
#             continue
#
#     except:
#         break

# In[10]:
# len(zombie_list)
# print(set(zombie_list))
# len(set(zombie_list))

# # In[12]:
#
#
# #谷歌2
# num=1
# for  web in set(zombie_list):
#     try:
#         if num%50 == 0:
#             print(num)
#
#         content = requests.get(web)
#         if content.status_code==200:
#             with open('E:\\图片\\谷歌恐怖血腥\\%d.jpg' %num, 'wb') as pic_fd:
#                 num+=1
#                 pic_fd.write(content.content)
#         else:
#             continue
#     except:
#         continue
#

# In[ ]:




