#coding=utf-8
import numpy as np

a=[4,1,2,54,7,8,9,3,4,546,62,4,234]
b=np.array(a)
c=np.argsort(-b)
print(c)