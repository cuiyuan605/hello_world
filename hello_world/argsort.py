#coding=utf-8
import numpy as np
d={'1':1,'2':2,'3':None}
d.pop('4','')
print(np.array(d.keys()))
print(np.array(d.values()))
a=[4,1,2,54,7,8,9,3,4,546,62,4,234]
b=np.array(a)
c=np.argsort(b)
print(c)
#print(reversed(b))
print(':'.join([str(x) for x in b]))
print(b[[x for x in reversed(c)]])
print([x for x in reversed(a)])