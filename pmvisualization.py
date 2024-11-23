import numpy as np
from vis import *
from datetime import datetime
from general_definitions import dt, N2

def timeToIndex(t):
    return abs(int(round(t/dt)) + N2)
idx = timeToIndex(31535800)

with open("r.npy", "rb") as file:
    r = np.load(file)

Sun = r[::50,0]
lign(Sun, '#fbb543')
Mercury = r[::50,1]
lign(Mercury, '#585858')
Venus = r[::50,2]
lign(Venus, '#b7711c')
Earth = r[:idx + 1:50,3]
lign(Earth, '#3d4782')
Mars = r[::50,4]
lign(Mars, '#ed795c')
Jupiter = r[::50,5]
lign(Jupiter, '#b8a48c')
Saturn = r[::50,6]
lign(Saturn, '#c4ad8d')
Uranus = r[::50,7]
lign(Uranus, '#c5ebee')
Neptune = r[::50,8]
lign(Neptune, '#497bfe')

idx = timeToIndex(31535800)
with open("rocketdata.npy", "rb") as file:
    rr = np.load(file)
lign(rr, 'black')

with open("testRsim2.npy", "rb") as file:
    rt = np.load(file)
lign(rt, "red")



point(r[timeToIndex(0), 3])
# point(r[timeToIndex(335573962.63948685), 5])
point(r[timeToIndex(31535800), 3])

m = max(r.min(), r.max(), key=abs)
setlim(m)

plot()
