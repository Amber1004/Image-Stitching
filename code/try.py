import numpy as np

arr = np.array([[0, 1, 0], [1, 0, 1]])
loc_a = np.array([1,2,3])
loc_b = np.array([4,5,6])
a, b = np.nonzero(arr)
# c =
print( [ (loc_a[i], loc_b[j]) for i,j in zip(a, b)])
print( [(i,j) for i,j in c])
print(loc_a[arr[0]])
