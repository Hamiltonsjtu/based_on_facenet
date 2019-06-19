import requests
import json
# #######-------------image client------------#######
files_ = {"file": open("F:/raw_image_web_crop_multi/maozedong/5.png", "rb")}

r = requests.post("http://0.0.0.0:5000/upload", files=files_)

returnval = json.loads(r.text)



