import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/gabriel/ma/0-Maturaarbeit')
from general_definitions import dt, N2
from math import ceil
#import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation

# -------  Importing the Data and necessary variables for usage ------- #
N2old = N2
N=12
N2 = int(N/2)
t0 = 373246737.99170226
t1 = 578499117.7331004
planetarydt = dt
dt = 100
sl = int(planetarydt/dt)

T = t1 - t0
steps = int(T/dt) + N2 + 2

b0 = t0 - dt*N2     #backpoint 0
IDX0 = ceil(b0/planetarydt) + N2old    #used later for slicing

IDX1 = int(t1/planetarydt) + N2old
p0 = ceil(b0/planetarydt)*sl - round(b0/dt)            #step in terms of dt - nearest calculated point = number of steps to be interpolated backwards & index of the first grid-point to be used
p1 = int(t1/planetarydt)*sl - round(b0/dt)              #from b0 to t1, how many points are used.

with open("/home/gabriel/ma/0-Maturaarbeit/saturnFinal.npy", "rb") as file:
    rsat = np.load(file)
rsat = rsat[p0:p1 + 1:sl]

with open("/home/gabriel/ma/0-Maturaarbeit/r.npy", "rb") as file:
    rpl = np.load(file)
rpl = rpl[IDX0:IDX1 + 1]

r = rpl[:,:-1]
r[:,-1] = rsat

def timeToIndex(t):
    a = np.int64(np.rint(t/planetarydt))
    return np.maximum(a, 0)

colorarray = np.array([[251,181,67],[88,88,88], [183,113,28], [61,71,130], [237,121,92], [184,164,140], [196,173,141],[0,0,0]], dtype=np.double)/255
def fade_line(r, colour):
    """
    Construct a 3d line with fading color
    """
    Npts = r.shape[0]
    # create colours array
    colours = np.zeros((Npts, 4))
    colours[:, 0:3] = colour
    colours[:, 3] = np.linspace(0, 1, Npts)

    # N-1 segments for N points
    # (x, y) start point and (x2, y2) end point
    # in 3D array each segment 2D inside
    segments = np.zeros((Npts-1, 2, 3))
    # segment start values - slice of last value
    segments[:, 0] = r[:-1]
    # segements end values - slice of first value
    segments[:,1] = r[1:]

    lc = Line3DCollection(segments, color=colours)
    return lc

fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

framerate = 24
daysPerSecond = 100
secondsPerFrame = daysPerSecond/framerate*24*60*60
ElementsToSkip = 200
orbitalPeriod = np.array([59800, 88.0, 224.7, 365.2, 687.0, 4331, 10747, 59800])*24*60*60

m = abs(max(r.min(), r.max(), key=abs))
ax.set_xlim(-m, m)
ax.set_ylim(-m, m)
ax.set_zlim(-m, m)
# plt.legend()
ax.set_axis_off()
ax.view_init(elev=90, azim=-70, roll=0)

for i in range(8):
    lc = fade_line(r[:2, i], colorarray[i])
    ax.add_collection(lc)
    ax.scatter(*r[1, i].T, s=0.7, color=colorarray[i])

def update(frame):
    UPPERINDEX = timeToIndex(frame*secondsPerFrame) + 1
    lowerINDEX = timeToIndex(frame*secondsPerFrame - orbitalPeriod)
    ax.clear()
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    ax.set_axis_off()
    ax.view_init(elev=90, azim=-70, roll=0)
    for i in range(8):
        ax.add_collection(fade_line(r[lowerINDEX[i]:UPPERINDEX:ElementsToSkip, i], colorarray[i]))
        ax.scatter(*r[UPPERINDEX - 1, i].T, s=0.7, color=colorarray[i], alpha=1)


firstTime = True
totalFrames = 0
lastFrame = 0
pbar = None
def callback(current_frame: int, total_frames: int):
    global totalFrames, pbar, firstTime, lastFrame
    if firstTime:
        firstTime = False
        totalFrames = total_frames
        pbar = tqdm(total=totalFrames)
    pbar.update(current_frame - lastFrame)
    lastFrame = current_frame

anim = animation.FuncAnimation(fig, update, frames=int(r.shape[0]*planetarydt/framerate), interval=framerate, blit=False)
# anim.save("animation1.mp4", progress_callback=callback)
# pbar.close()
plt.show()
