import numpy as np
import time
a=np.array([1,2,3,4,])
print(a)
# [1 2 3 4]
a=np.random.rand(1000000)
b=np.random.rand(1000000)
tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(a)
print("vsctorized version:"+str(1000*(toc-tic))+"ms")
c=0
tic=time.time()
for i in range(1000000):
    c +=a[i]*b[i]
toc=time.time()
print(c)
print("vsctorized version:"+str(1000*(toc-tic))+"ms")

import numpy as np

arr1 = np.array([[0, 0, 0],[1, 1, 1],[2, 2, 2], [3, 3, 3]])  #arr1.shape = (4,3)
arr2 = np.array([[1],[2],[3],[4]])    #arr2.shape = (4, 1)

arr_sum = np.dot(arr1,arr2)
print(arr_sum)