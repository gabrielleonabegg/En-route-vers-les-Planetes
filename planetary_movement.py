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
print("downloaded data from horizons api")

def ft(posvects: np.ndarray, n: int):
    a = np.array([0,0,0], dtype=np.longdouble)
    for i in range(objcount):
        if i != n:
            rl = posvects[i] - posvects[n]   #make sure that the local r doesn't override the global one
            if mag(rl) == 0:
                print("Don't divide by 0 you idiot!")
            a += rl*G*(masses[i])/mag(rl)**3
    return a

#for all planets: calculate m1 and k1, use the vectors obtained to recalculate the force field
#then on the basis of this, calculate m2 and k2 and so on.

#for the startup procedure of the rocket, we need to have the intermediate positions of the planets:
startup = np.empty([N + 1, 3 , objcount, 3], dtype=np.longdouble)

stepli = np.empty([4,2,objcount,3], dtype=np.longdouble)

#how to access r and v for a given step:
    # r = r[N2 - k]
    # v = v[N2 - k]
dt = -dt
for k in range(N2):
    stepli[0][0] = v[N2 - k]                    #[0][0] represents m and therefore a velocity
    for i in range(objcount):
        stepli[0][1][i] = ft(r[N2 - k], i)      #[0][1] represents k (rungekutta.py) and therefore an acceleration

    for i in range(1,4):
        #calculate m
        for j in range(objcount):
            if i < 3:
                stepli[i][0][j] = v[N2 - k][j] + stepli[i - 1][1][j]*dt/2     #m1 & m2 (see rungekutta.py)
            else:
                stepli[i][0][j] = v[N2 - k][j] + stepli[i - 1][1][j]*dt
        #calculate k (see rungekutta.py), has nothing to do with the local for-loop variable k
        # instead of recalculating this vector for every mass, it is saved in stepvectm
        if i < 3:
            stepvectm = r[N2 - k] + stepli[i][0]*dt/2
        else:
            stepvectm = r[N2 - k] + stepli[i][0]*dt
        startup[N2 - k - 1][i - 1] = stepvectm
        for j in range(objcount):        #only after calculating all m's can we calculate all k's
            stepli[i][1][j] = ft(stepvectm, j)
    #finally calculate new r and v
    r[N2 - k - 1] = r[N2 - k] + (stepli[0][0] + stepli[1][0]*2 + stepli[2][0]*2 + stepli[3][0])*dt/6
    v[N2 - k - 1] = v[N2 - k] + (stepli[0][1] + stepli[1][1]*2 + stepli[2][1]*2 + stepli[3][1])*dt/6

dt = -dt
for k in range(N2):
    stepli[0][0] = v[N2 + k]                    #[0][0] represents m and therefore a velocity
    for i in range(objcount):
        stepli[0][1][i] = ft(r[N2 + k], i)      #[0][1] represents k (rungekutta.py) and therefore an acceleration

    for i in range(1,4):
        #calculate m
        for j in range(objcount):
            if i < 3:
                stepli[i][0][j] = v[N2 + k][j] + stepli[i - 1][1][j]*dt/2     #m1 & m2 (see rungekutta.py)
            else:
                stepli[i][0][j] = v[N2 + k][j] + stepli[i - 1][1][j]*dt
        #calculate k (see rungekutta.py), has nothing to do with the local for-loop variable k
        # instead of recalculating this vector for every mass, it is saved in stepvectm
        if i < 3:
            stepvectm = r[N2 + k] + stepli[i][0]*dt/2
        else:
            stepvectm = r[N2 + k] + stepli[i][0]*dt
        startup[N2 + k][i - 1] = stepvectm
        for j in range(objcount):        #only after calculating all m's can we calculate all k's
            stepli[i][1][j] = ft(stepvectm, j)
    #finally calculate new r and v
    r[N2 + k + 1] = r[N2 + k] + (stepli[0][0] + stepli[1][0]*2 + stepli[2][0]*2 + stepli[3][0])*dt/6
    v[N2 + k + 1] = v[N2 + k] + (stepli[0][1] + stepli[1][1]*2 + stepli[2][1]*2 + stepli[3][1])*dt/6
print("Starting procedure complete")

#now that the startvalues are initialized, we no longer need stepli and can free its memory:
stepli = None
del stepli

#the ordinate coefficients are defined in general_definitions

C1s = np.empty([objcount,3], dtype=np.longdouble)
S0 = np.empty([objcount,3], dtype=np.longdouble)

def resets():
    global C1s, S0, Sn, sn
    #defining C1s
    for i in range(objcount):
        sum1 = np.array([0,0,0], dtype=np.longdouble)
        for k in range(N + 1):
            sum1 += ft(r[k], i)*b[N2][k]
        C1s[i] = v[N2][i]/dt - sum1
    #Defining S0:
    for i in range(objcount):
        sum2 = np.array([0,0,0], dtype=np.longdouble)
        for k in range(N + 1):
            sum2 += ft(r[k], i)*a[N2][k]
        S0[i] = r[N2][i]/dt**2 - sum2
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
        for i in range(objcount):
            for j in range(N2 - n):
                Sn[i] = Sn[i] - sn[i] + ft(r[N2 - j], i)*0.5
                sn[i] -= (ft(r[N2 - j], i) + ft(r[N2 - j - 1], i))*0.5
        return sn, Sn
    elif n > N2:
        resets()
        for i in range(objcount):
            for j in range(n - N2):
                Sn[i] += sn[i] + ft(r[N2 + j], i)*0.5
                sn[i] += (ft(r[N2 + j], i) + ft(r[N2 + j + 1], i))*0.5
        return sn, Sn

a_1 = np.empty_like(sn)
def getsr(n):
    global Sn, sn, a_1
    if n == N + 1:
        resets()
        for i in range(len(masses)):
            for j in range(n - N2 - 1):
                a_1[i] = ft(r[N2 + j], i)
                if j != 0:
                    sn[i] += (ft(r[N2 + j - 1], i) + a_1[i])*0.5
                Sn[i] += sn[i] + a_1[i]*0.5
    for i in range(len(masses)):
        a_1[i] = ft(r[n - 1], i)
        sn[i] += (ft(r[n - 2], i) + a_1[i])*0.5
        Sn[i] += sn[i] + a_1[i]*0.5
    return sn, Sn

def getssr(n):
    global sn, a_1
    cpy = sn.copy()
    for i in range(len(masses)):
        cpy[i] += (a_1[i] + ft(r[n], i))*0.5
    return cpy
#Correct starting values!!!!!!!!!!

#while max change in acceleration is higher than ...
#for all points:
    #for all planets:
        #apply corrector
#get max change in acceleration

maxa = 1
while maxa > 0.00000000001:
    maxa = 0
    for n in range(N + 1):
        if n != N2:
            s, S = getss(n)
            oldr = r[n]
            for o in range(len(masses)):
                sum3r = np.array([0,0,0], dtype=np.longdouble)
                sum3v = np.array([0,0,0], dtype=np.longdouble)
                for k in range(N + 1):
                    ao = ft(r[k], o)
                    sum3r += ao*a[n][k]
                    sum3v += ao*b[n][k]
                r[n][o] = (S[o] + sum3r)*dt**2
                v[n][o] = (s[o] + sum3v)*dt
            for o in range(len(masses)):
                aold = ft(oldr, o)
                anew = ft(r[n], o)
                magdif = mag(aold - anew)
                if magdif > maxa:
                    maxa = magdif


#Commencing PEC cycle:
n = N
t = N2*dt

corrsum = np.empty([2, objcount,3], dtype=np.longdouble)        #corrsum[0]: position, corrsum[1]: velocity
with tqdm(total=steps-9) as pbar:
    while t <= T:        #T is defined in general_definitions
        #Predict:
        s, S = getsr(n + 1)         #returns sn and Sn+1, sn is used for the predictor
        for o in range(objcount):
            psumr = np.array([0,0,0], dtype=np.longdouble)
            psumv = np.array([0,0,0], dtype=np.longdouble)
            for k in range(N + 1):
                pa = ft(r[n-N+k], o)
                psumr += pa*a[N + 1][k]
                psumv += pa*b[N + 1][k]
            r[n + 1][o] = (psumr + S[o])*dt**2
            v[n + 1][o] = (s[o] + ft(r[n], o)/2 + psumv)
        n += 1
        corrsum.fill(0)
        #Evaluate-Correct:# May not make a difference for Gauss-Jackson, but summed Adams becomes unstable very quickely when not corrected.
        for o in range(objcount):
            for k in range(N):
                ac = ft(r[n + k - N], o)
                corrsum[0][o] += ac*a[N][k]
                corrsum[1][o] += ac*b[N][k]
        for _ in range(200):
            max = 0
            s = getssr(n)
            for o in range(objcount):
                rold = r[n][o]
                vold = v[n][o]
                aco = ft(r[n], o)
                r[n][o] = (aco*a[N][N] + corrsum[0][o] + S[o])*dt**2
                v[n][o] = (aco*b[N][N] + corrsum[1][o] + s[o])*dt
                diff = mag(rold - r[n][o])
                diffv = mag(vold - v[n][o])
                if diff > max:
                    max = diff
                if diffv > max:
                    max = diffv
            if max < 0.0000000001:
                break
        t += dt
        pbar.update(1)

print("data calculated")
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