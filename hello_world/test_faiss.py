#coding=utf-8
import os
import faiss
import numpy as np
from datetime import datetime
import time
import pickle
from sklearn import preprocessing

# 获取商品图片的特征向量
def get_feature_vec(dir_path,item_id):
    vec=[]
    product_dir=os.path.join(dir_path,item_id)
    feature_path=os.path.join(product_dir,item_id+".pkl")
    if not os.path.exists(feature_path):
        return vec
    with open(feature_path, 'rb') as f:
        vec = pickle.load(f)
    vec_length=1
    for d in vec.shape:
        vec_length*=d
    reshaped_vec=vec.detach().reshape(1,vec_length)
    return reshaped_vec

feature_dir_path="/share/code/cuiy/SimilarImage/image_features/女鞋/低帮鞋/"
item_id_list=[]
item_vec_list=[]
d=4032
k=30

print("reading feature files.....")
for item_id in os.listdir(feature_dir_path):
    vec=get_feature_vec(feature_dir_path,item_id)
    if len(vec)<1:
        continue
    vec_normalized=preprocessing.normalize(vec, norm='l2')
    # faiss need float32
    vec_normalized=np.squeeze(vec_normalized).astype('float32')
    d=len(vec_normalized)
    item_id_list.append(item_id)
    item_vec_list.append(vec_normalized)
item_vec_array=np.array(item_vec_list)

'''
d = 4032
nb = 60
nq = 10
k = 30
rs = np.random.RandomState(123)
xb = rs.rand(nb, d).astype('float32')
xq = rs.rand(nq, d).astype('float32')
'''

print("d:",d)
index_cpu = faiss.IndexFlatIP(d)

assert faiss.get_num_gpus() > 1

start_time=time.time()
co = faiss.GpuMultipleClonerOptions()
co.shard = True
index = faiss.index_cpu_to_all_gpus(index_cpu, co, ngpu=2)
end_time=time.time()
print("init index:",end_time-start_time)

start_time=time.time()
index.add(item_vec_array)
D, I = index.search(item_vec_array, k)
end_time=time.time()
print("gpu search:",end_time-start_time)
print("idx:",I)
print("dist:",D)

'''
start_time=time.time()
index_cpu.add(item_vec_array)
D_ref, I_ref = index_cpu.search(item_vec_array, k)
print()
end_time=time.time()
print("cpu search:",end_time-start_time)
print("idx:",I)
print("dist:",D)

assert np.all(I == I_ref)
'''

del index
index2 = faiss.index_cpu_to_all_gpus(index_cpu, co, ngpu=2)
index.clean()
D2, I2 = index2.search(item_vec_array, k)
print("idx:",I2)
print("dist:",D2)