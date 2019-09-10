import pickle
import numpy as np
from beautifultable import BeautifulTable

def distance(embeddings1, embeddings2):
    dot = np.dot(embeddings1, embeddings2)
    norm = np.linalg.norm(embeddings1, ord=2) * np.linalg.norm(embeddings2, ord=2)
    similarity = dot / norm
    if similarity > 1:
        similarity = 1.0
    return similarity


with open('./embds.pkl', 'rb') as f:
    data = pickle.load(f)
dist = np.zeros((len(data), len(data)))
data_keys = list(data.keys())


for i in range(len(data_keys)):
    for j in range(len(data_keys)):
        embedding_1 = data[data_keys[i]]
        embedding_2 = data[data_keys[j]]
        dist[i,j] = distance(embedding_1, embedding_2)

table = BeautifulTable()
table.column_headers = data_keys
for i in range(len(data_keys)):
    table.append_row(dist[i,:])
print(table)
# print('face :{}-->{} similarity: {} '.format(data_keys[i], data_keys[j], dist[i,j]))

