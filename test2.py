from kepler import kepler
from helpers1 import G, mag
import numpy as np
from horizons import horizons
from general_definitions import objcount, masses, dt, N2
import general_definitions
import sys

def timeToIndex(t):
    return int(round(t/dt)) + N2

try:
    T = float(sys.argv[1])
    print(T)

    if T > general_definitions.T:
        print("Time t larger than specified in general_definitions\nTerminating program")
        sys.exit(1)
    elif T < 0:
        print("No negative values accepted")
        sys.exit(1)

    n = timeToIndex(T)
    #read into memory position vectors:
    #Gauss-Jackson
    print(n)
    with open("r.npy", "rb") as file:
        myr = np.load(file)
        myr = myr[n]
    #read into memory velocity vectors for comparison:
    with open("v.npy", "rb") as file:
        myv = np.load(file)
        myv = myv[n]
except IndexError:
    from general_definitions import T
    with open("r.npy", "rb") as file:
        myr = np.load(file)
        myr = myr[-1]
    #read into memory velocity vectors for comparison:
    with open("v.npy", "rb") as file:
        myv = np.load(file)
        myv = myv[-1]


def direction(r:np.ndarray):
    #get solely the direction of the vectors:
    direction = np.empty([len(r), 3], dtype=np.longdouble)
    for i in range(len(r)):
        direction[i] = r[i]/mag(r[i])
    return direction

def dist(r:np.ndarray):
    #calculate the distance from the sun:
    dist = np.empty([len(r)], dtype=np.longdouble)
    for i in range(objcount):
        dist = mag(r[i])
    return dist

def heliocentric(r:np.ndarray):
    #calculate position with respect to the sun
    helior = np.empty([objcount - 1,3], dtype=np.longdouble)
    for i in range(objcount - 1):
        helior[i] = r[i + 1] - r[0]
    return helior

def error(r1:np.ndarray, r2:np.ndarray):
    return "{:.4}".format(mag(r1 - r2)/mag(r1)*100) + " \%"

def errdist(r1:np.ndarray, r2:np.ndarray):
    return "{:.4}".format(abs(mag(r1) - mag(r2))/mag(r1)*100) + " \%"

"""
#read into memory position vectors:
#Gauss-Jackson
with open("r.npy", "rb") as file:
    myr = np.load(file)
    myr = myr[-1]
#read into memory velocity vectors for comparison:
with open("v.npy", "rb") as file:
    myv = np.load(file)
    myv = myv[-1]
"""


#JPL's Horizons:
horizonsr = np.empty([objcount,3], dtype=np.longdouble)
horizonsv = np.empty([objcount,3], dtype=np.longdouble)
horizonsr[0], horizonsv[0] = horizons(10, T)
for i in range(objcount - 1):
    horizonsr[i + 1], horizonsv[i + 1] = horizons(i + 1,  T)

#Kepler Problem:
keplerr = np.empty([objcount - 1,3], dtype=np.longdouble)
keplerv = np.empty([objcount - 1,3], dtype=np.longdouble)
helios_r, helios_v = horizons(10, 0)
for i in range(objcount - 1):
    r1, v1 = horizons(i + 1, 0)
    keplerr[i], keplerv[i] = kepler(r1 - helios_r, v1 - helios_v, T, (masses[0] + masses[i + 1])*G)

#Idea with the origin
print("Test new origin idea : ", end="")
bary = np.array([0,0,0], dtype=np.longdouble)
for i in range(objcount):
    bary += masses[i] * myr[i]
print(bary)
print("As a reference the same calculation for horizons: ", end="")
bary.fill(0)
for i in range(objcount):
    bary += masses[i] * horizonsr[i]
print(bary)

#convert to heliocentric coodrinates:
myr = heliocentric(myr)
myv = heliocentric(myv)
horizonsr = heliocentric(horizonsr)
horizonsv = heliocentric(horizonsv)

print("PM to Horizons: ")
for i in range(objcount - 1):
    print(error(horizonsr[i], myr[i]))
print("Kepler to Horizons: ")
for i in range(objcount - 1):
    print(error(horizonsr[i], keplerr[i]))
print("Kepler to PM")
for i in range(objcount - 1):
    print(error(myr[i], keplerr[i]))

print("Distance; PM to Horizons: ")
for i in range(objcount - 1):
    print(errdist(myr[i], horizonsr[i]))
print("Distance; Kepler to Horizons: ")
for i in range(objcount - 1):
    print(errdist(keplerr[i], horizonsr[i]))
print("Distance; Kepler to PM: ")
for i in range(objcount - 1):
    print(errdist(keplerr[i], myr[i]))

print("Velocity tests :\n\n")
print("PM to Horizons: ")
for i in range(objcount - 1):
    print(error(horizonsv[i], myv[i]))
print("Kepler to Horizons: ")
for i in range(objcount - 1):
    print(error(horizonsv[i], keplerv[i]))
print("Kepler to PM")
for i in range(objcount - 1):
    print(error(myv[i], keplerv[i]))

print("Distance; PM to Horizons: ")
for i in range(objcount - 1):
    print(errdist(horizonsv[i], myv[i]))
print("Distance; Kepler to Horizons: ")
for i in range(objcount - 1):
    print(errdist(horizonsv[i], keplerv[i]))
print("Distance; Kepler to PM: ")
for i in range(objcount - 1):
    print(errdist(myv[i], keplerv[i]))