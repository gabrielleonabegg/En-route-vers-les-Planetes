import numpy as np
from helpers1 import *
from horizons import horizons
from vis import *
from general_definitions import *
from tqdm import tqdm

r = np.empty([steps,9,3], dtype=np.longdouble)
v = np.empty([steps,9,3], dtype=np.longdouble)

rinit, vinit = horizons(10)
r[N2][0] = rinit
v[N2][0] = vinit

#get initial conditions:
for i in range(8):
    rh, vh = horizons(i + 1)
    r[N2][i + 1] = rh
    v[N2][i + 1] = vh
print("Download from Horizons API complete!")

masses1 = masses[np.newaxis, :, np.newaxis]
def aG(rAllPlanets):
    deltar = rAllPlanets[np.newaxis, :] - rAllPlanets[:,np.newaxis]
    distanceFactor = (np.sqrt(np.sum(deltar**2, axis=-1))**3)[:,:,np.newaxis]
    return np.sum(np.divide(deltar, distanceFactor, out=np.zeros_like(deltar), where=np.rint(distanceFactor)!=0)*G*masses1, axis=1)

masses2 = masses[np.newaxis, np.newaxis, :, np.newaxis]
def aGeneral2(rAllPlanets):
    deltar = rAllPlanets[:,np.newaxis, :] - rAllPlanets[:,:,np.newaxis]
    distanceFactor = (np.sqrt(np.sum(deltar**2, axis=-1))**3)[:,:,:,np.newaxis]
    return np.sum(np.divide(deltar, distanceFactor, out=np.zeros_like(deltar), where=np.rint(distanceFactor)!=0)*G*masses1, axis=2)


stepli = np.empty([4,2,objcount,3], dtype=np.longdouble)

dt = -dt
for k in range(N2):
    stepli[0, 0] = v[N2 - k]
    stepli[0, 1] = aG(r[N2 - k])
    stepli[1, 0] = v[N2 - k] + stepli[0, 1]*dt/2
    stepli[1, 1] = aG(r[N2 - k] + stepli[0, 0]*dt/2)
    stepli[2, 0] = v[N2 - k] + stepli[1, 1]*dt/2
    stepli[2, 1] = aG(r[N2 - k] + stepli[1, 0]*dt/2)
    stepli[3, 0] = v[N2 - k] + stepli[2, 1]*dt
    stepli[3, 1] = aG(r[N2 - k] + stepli[2, 0]*dt)
    r[N2 - k - 1] = r[N2 - k] + (stepli[0, 0] + stepli[1, 0]*2 + stepli[2, 0]*2 + stepli[3, 0])*dt/6
    v[N2 - k - 1] = v[N2 - k] + (stepli[0, 1] + stepli[1, 1]*2 + stepli[2, 1]*2 + stepli[3, 1])*dt/6

dt = -dt
for k in range(N2):
    stepli[0, 0] = v[N2 + k]
    stepli[0, 1] = aG(r[N2 + k])
    stepli[1, 0] = v[N2 + k] + stepli[0, 1]*dt/2
    stepli[1, 1] = aG(r[N2 + k] + stepli[0, 0]*dt/2)
    stepli[2, 0] = v[N2 + k] + stepli[1, 1]*dt/2
    stepli[2, 1] = aG(r[N2 + k] + stepli[1, 0]*dt/2)
    stepli[3, 0] = v[N2 + k] + stepli[2, 1]*dt
    stepli[3, 1] = aG(r[N2 + k] + stepli[2, 0]*dt)
    r[N2 + k + 1] = r[N2 + k] + (stepli[0, 0] + stepli[1, 0]*2 + stepli[2, 0]*2 + stepli[3, 0])*dt/6
    v[N2 + k + 1] = v[N2 + k] + (stepli[0, 1] + stepli[1, 1]*2 + stepli[2, 1]*2 + stepli[3, 1])*dt/6

#now that the startvalues are initialized, we no longer need stepli and can free its memory:
stepli = None
del stepli

C1s = np.empty([objcount,3], dtype=np.longdouble)
S0 = np.empty([objcount,3], dtype=np.longdouble)

def ft(posvects: np.ndarray, n: int):
    a = np.array([0,0,0], dtype=np.longdouble)
    for i in range(objcount):
        if i != n:
            rl = posvects[i] - posvects[n]   #make sure that the local r doesn't override the global one
            if mag(rl) == 0:
                print("Don't divide by 0 you idiot!")
            a += rl*G*(masses[i])/mag(rl)**3
    return a


def resets():
    global C1s, S0, Sn, sn
    #defining C1s
    C1s[:] = v[N2]/dt - np.sum(aGeneral2(r[:N + 1]) * b[N2][:,np.newaxis, np.newaxis], axis=0)
    #Defining S0:
    S0[:] = r[N2]/dt**2 - np.sum(aGeneral2(r[:N + 1]) * a[N2][:,np.newaxis, np.newaxis], axis=0)
    sn = C1s        #since S0 and C1s never actually get used, passing on the pointer to the array instead of copying it doesn't constitute a bug
    Sn = S0
resets()

def getss(n):
    global Sn, sn
    if n == N2:
        resets()
        return Sn
    elif -1 < n < N2:
        resets()
        for i in range(N2 - n):
            Sn = Sn - sn + aG(r[N2 - i])*0.5
            sn -= (aG(r[N2 - i]) + aG(r[N2 - i - 1]))*0.5
        return sn, Sn
    elif n > N2:
        resets()
        for i in range(n - N2):
            Sn += sn + aG(r[N2 + i])*0.5
            sn += (aG(r[N2 + i]) + aG(r[N2 + i + 1]))*0.5
        return sn, Sn

a_1 = np.empty_like(sn)
def getsr(n):
    global Sn, sn, a_1
    if n == N + 1:
        resets()
        for j in range(n - N2 - 1):
            a_1 = aG(r[N2 + j])
            if j != 0:
                sn += (aG(r[N2 + j - 1]) + a_1)*0.5
            Sn += sn + a_1*0.5
    a_1 = aG(r[n - 1])
    sn += (aG(r[n - 2]) + a_1)*0.5
    Sn += sn + a_1*0.5
    return sn, Sn

def getssr(n):
    global sn, a_1
    cpy = sn.copy()
    cpy += (a_1 + aG(r[n]))*0.5
    return cpy

maxa = 1
while maxa > 0.00000000001:
    maxa = 0
    for n in range(N + 1):
        if n != N2:
            s, S = getss(n)
            oldr = r[n]
            a0 = aGeneral2(r[:N + 1])
            sum3r = np.sum(a0 * a[n][:,np.newaxis, np.newaxis], axis=0)
            sum3v = np.sum(a0 * b[n][:,np.newaxis, np.newaxis], axis=0)
            r[n] = (S + sum3r)*dt**2
            v[n] = (s + sum3v)*dt
            for o in range(len(masses)):
                aold = aG(oldr)
                anew = aG(r[n])
                maxa = np.max(mag(aold - anew))

#Commencing PEC cycle:
n = N
t = N2*dt

corrsum = np.empty([2, objcount,3], dtype=np.longdouble)        #corrsum[0]: position, corrsum[1]: velocity
with tqdm(total=steps-9) as pbar:
    while t <= T:        #T is defined in general_definitions
        #Predict:
        s, S = getsr(n + 1)         #returns sn and Sn+1, sn is used for the predictor
        pa = aGeneral2(r[n - N:n + 1])
        psumr = np.sum(pa * a[N + 1][:,np.newaxis, np.newaxis], axis=0)
        psumv = np.sum(pa * b[N + 1][:,np.newaxis, np.newaxis], axis=0)
        r[n + 1] = (psumr + S)*dt**2
        v[n + 1] = (s + aG(r[n])/2 + psumv)
        n += 1
        corrsum.fill(0)
        #Evaluate-Correct:# May not make a difference for Gauss-Jackson, but summed Adams becomes unstable very quickely when not corrected.
        ac = aGeneral2(r[n - N:n])
        corrsum[0] = np.sum(ac * a[N][:-1,np.newaxis, np.newaxis], axis=0)
        corrsum[1] = np.sum(ac * b[N][:-1,np.newaxis, np.newaxis], axis=0)

        for _ in range(200):
            max = 0
            s = getssr(n)
            rold = r[n]
            vold = v[n]
            aco = aG(r[n])
            r[n] = (aco*a[:,:, np.newaxis, np.newaxis][N, N] + corrsum[0] + S)*dt**2
            v[n] = (aco*b[:,:, np.newaxis, np.newaxis][N, N] + corrsum[1] + s)*dt
            maxr = np.max(mag(rold - r[n]))
            maxv = np.max(mag(vold - v[n]))
            if maxr < 0.0000000001 and maxv < 0.0000000001:
                break
        t += dt
        pbar.update(1)

print("Data calculated")
print(t)

with open("r.npy", "wb") as file:   #'wb': write as binary
    np.save(file, r)
with open("v.npy", "wb") as file:
    np.save(file, v)

epoch = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
with open("log.txt", "w") as file:
    file.write(str(epoch.timestamp()))

import os
os.system("shutdown.exe /s")