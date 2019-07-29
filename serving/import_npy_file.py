
import numpy as np

emb_data = np.load('people_embs_V1.npy').item()

for key in emb_data:
    print(key)
    print(emb_data[key])

print(type(emb_data))
