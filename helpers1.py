from math import sqrt, pi
import numpy as np

G = np.longdouble(6.6743015e-11)

def vect(a,b,c):
    return np.array([a,b,c], dtype=np.longdouble)

def mag(array:np.ndarray):
    if array.shape[-1] != 3:
        print("Warning: An array of 3D vectors instead of {}D vectors expected".format(array.shape[-1]))
    return np.sqrt(np.sum(array*array, axis=-1))

def unsignedAngle(array1:np.ndarray, array2:np.ndarray):    #takes two arrays of vectors as inputs, speeds up work with large datasets as we don't need a for loop (about 60 times faster for large arrays, but since both operations are rather fast, we don't actually care)
    if array1.shape != array2.shape:
        print("Arrays are not of the same shape")
        return None
    if array1.shape[-1] != 3:
        print("Warning: 3-dimentional vectors are expected")
    dot = np.sum(array1*array2, axis=-1)
    mag = np.sqrt(np.sum(array1**2, -1)*np.sum(array2**2, axis=-1))
    return np.arccos(dot/mag)
class object:
    def __init__(self, r: vect, v: vect, m):
        self.r = r
        self.v = v
        self.m = m

class planet(object):
    def __init__(self, r: vect, v: vect, m, rayon):
        super().__init__(r, v, m)
        self.rayon = rayon

def norm(v):
    return v/mag(v)