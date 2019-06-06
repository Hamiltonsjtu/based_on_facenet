import requests
# #######-------------image client------------#######
files_ ={"file": open("e:/shuai/Face/hu_ying.jpg" ,"rb")}
r = requests.post("http://127.0.0.1:5000/upload", files=files_)
print(r.text)