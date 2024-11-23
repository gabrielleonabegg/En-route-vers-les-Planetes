# [1]: https://www.youtube.com/watch?v=fAztJg9oi7s
# [2]: https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html#sphx-glr-gallery-mplot3d-lorenz-attractor-py
# [3]: https://stackoverflow.com/questions/45148704/how-to-hide-axes-and-gridlines
# [4]: https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
ax = plt.axes(projection="3d")
ax.set_aspect(aspect="equal")
ax.set_box_aspect(aspect=(1,1,1))

def reset():
    global ax
    ax = plt.axes(projection="3d")
    ax.set_aspect(aspect="equal")

def point(pos:np.ndarray, col:str = 'g'):
    x,y,z = pos
    ax.scatter([x], [y], [z], color=col, s=2)

def lign(arr:np.ndarray, col:str):
    ax.grid(False)
    ax.set_box_aspect([1,1,1])
    ax.plot(*arr.T, color=col, linewidth='0.5')     #[2]

def sphere(pos:np.ndarray, r, col:str):
    x,y,z = pos
    print(x,y,z)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
    x = np.cos(u)*np.sin(v)*r + x
    y = np.sin(u)*np.sin(v)*r + y
    z = np.cos(v)*r + z
    ax.plot_wireframe(x, y, z, color=col)

def setlim(m):
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)

def plot():
    ax.set_aspect(aspect="equal")
    ax.set_box_aspect(aspect=(1,1,1))
    plt.show()