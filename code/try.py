import numpy as np
import math


arr = np.array([[0, 1, 0], [1, 0, 1]])
loc_a = np.array([1,2,3.5])
loc_b = np.array([4,5,6])
a, b = np.nonzero(arr)
# c =
# print( [ (loc_a[i], loc_b[j]) for i,j in zip(a, b)])
# print( [(i,j) for i,j in c])
# print(loc_a[arr[0]])
print(math.ceil(np.max(loc_a)))
print((arr[0] + arr[1])//2)


