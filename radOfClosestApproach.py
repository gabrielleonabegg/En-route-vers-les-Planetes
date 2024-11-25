import numpy as np
# radius = np.array([2440000.53, 6051000.893, 6378000.137, 3396000.19, 71492000, 60268000, 25559000, 24766000], dtype=np.longdouble)
#Vénus, Mars, Uranus, Neptune
p0 = np.array([65, 0.020, 0.42, 0.45], dtype=np.longdouble)
H = np.array([15.9, 11.1, 27.7, 20.3], dtype=np.longdouble)*1000
re = np.array([6051800, 3396200, 25559000, 24764000], dtype=np.longdouble)

p = 2.438e-8*np.exp(-2/9.473)

print(p)
def rds(re, p0, H):
    return re - H*(np.log(p) - np.log(p0))

print(rds(re, p0, H))