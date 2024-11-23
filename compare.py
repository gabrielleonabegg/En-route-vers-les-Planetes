import numpy as np
from helpers1 import mag, G
from general_definitions import masses
from tqdm import tqdm
from math import sqrt
from time import time
mu = G*masses[3]
def f(r):
    return - mu/mag(r)**3 * r


r0 = np.array([6478100,0, 0], dtype=np.longdouble)
v0 = np.array([0, 10000, 0], dtype=np.longdouble)

E = 0.5 * mag(v0)**2 - mu/mag(r0)
h = np.cross(r0, v0)
a = - mu/(2*E)
p = mag(h)**2/(mu)
e = sqrt(1+(2*E*mag(h)**2)/(mu)**2)
rp = p/(1+e)
ra = p/(1 - e)
# Diskussion der Daten: Siehe Arbeitsjournal

dt = 0.05     #0.1
TP = 2*np.pi*pow(a, 3/2)/sqrt(mu)
nrev = 100

# stepn = int(TP*nrev/dt + 2)

# r = np.empty([stepn,3], dtype=np.longdouble)
# r[0] = r0
# v = np.empty([stepn,3], dtype=np.longdouble)
# v[0] = v0


# start = time()
# i = 0
# with tqdm(total=(stepn - 1)) as pbar:
#     while i*dt < TP*nrev:
#         a = f(r[i])
#         r[i + 1] = r[i] + v[i]*dt
#         v[i + 1] = v[i] + a*dt
#         i += 1
#         pbar.update(1)

# print("Time elapsed {}s".format(time() - start))
# print(r[-1])
# print("Forward Euler with step-size {}s after {} revolutions has an imprecision of {}% & {}%".format(dt, nrev, mag(r[-1]-r0)/mag(r0), mag(v[-1]-v0)/mag(v0)))

#Runge-Kutta du quatrième ordre :

dt = 6                     #1
stepn = int(TP*nrev/dt + 2)

r = np.empty([stepn,3], dtype=np.longdouble)
r[0] = r0
v = np.empty([stepn,3], dtype=np.longdouble)
v[0] = v0

start = time()
i = 0
with tqdm(total=(stepn - 1)) as pbar:
    while i*dt < TP*nrev:
        m0 = v[i]
        k0 = f(r[i])
        m1 = v[i] + k0*dt/2
        k1 = f(r[i] + m0*dt/2)
        m2 = v[i] + k1*dt/2
        k2 = f(r[i] + m1*dt/2)
        m3 = v[i] + k2*dt
        k3 = f(r[i] + m2*dt)
        r[i + 1] = r[i] + ((m0+ 2*m1+ 2*m2+ m3)*dt/6)
        v[i + 1] = v[i] + (k0 + k1*2 + k2*2 + k3)*dt/6
        i+=1
        pbar.update(1)
print("Time elapsed {}s".format(time() - start))
print(r[-1])
print("RK4 with step-size {}s after {} revolutions has an imprecision of {}% & {}%".format(dt, nrev, mag(r[-1] - r0)/mag(r0), mag(v[-1] - v0)/mag(v0)))



from general_definitions import N2, N, a, b

#Gauss-Jackson avec 

dt = 30
stepn = int(TP*nrev/dt + 2)


r = np.empty([stepn + N2,3], dtype=np.longdouble)
r[N2] = r0
v = np.empty([stepn + N2,3], dtype=np.longdouble)
v[N2] = v0


def fs(r):
    return - mu/mag(r)**3 * r

def f(n):
    return - mu/mag(r[n])**3 * r[n]




dt = -dt
for k in range(N2):
    m0 = v[N2 - k]
    k0 = fs(r[N2 - k])
    m1 = v[N2 - k] + k0*dt/2
    k1 = fs(r[N2 - k] + m0*dt/2)
    m2 = v[N2 - k] + k1*dt/2
    k2 = fs(r[N2 - k] + m1*dt/2)
    m3 = v[N2 - k] + k2*dt
    k3 = fs(r[N2 - k] + m2*dt)
    r[N2 - k - 1] = r[N2 - k] + (m0 + m1*2 + m2*2 + m3)*dt/6
    v[N2 - k - 1] = v[N2 - k] + (k0 + k1*2 + k2*2 + k3)*dt/6

dt = -dt
for k in range(N2):
    m0 = v[N2 + k]
    k0 = fs(r[N2 + k])
    m1 = v[N2 + k] + k0*dt/2
    k1 = fs(r[N2 + k] + m0*dt/2)
    m2 = v[N2 + k] + k1*dt/2
    k2 = fs(r[N2 + k] + m1*dt/2)
    m3 = v[N2 + k] + k2*dt
    k3 = fs(r[N2 + k] + m2*dt)
    r[N2 + k + 1] = r[N2 + k] + (m0 + m1*2 + m2*2 + m3)*dt/6
    v[N2 + k + 1] = v[N2 + k] + (k0 + k1*2 + k2*2 + k3)*dt/6

C1s = np.empty([3], dtype=np.longdouble)
S0 = np.empty([3], dtype=np.longdouble)

def resets():
    global C1s, S0, Sn, sn
    #defining C1s
    sum1 = np.array([0,0,0], dtype=np.longdouble)
    for k in range(N + 1):
        sum1 += f(k)*b[N2][k]
    C1s = v[N2]/dt - sum1
    #Defining S0:
    sum2 = np.array([0,0,0], dtype=np.longdouble)
    for k in range(N + 1):
        sum2 += f(k)*a[N2][k]
    S0 = r[N2]/dt**2 - sum2
    sn = C1s
    Sn = S0
resets()

def getss(n):
    global Sn, sn
    if n == N2:
        resets()
        return Sn
    elif -1 < n < N2:
        resets()
        for j in range(N2 - n):
            Sn = Sn - sn + f(N2 - j)*0.5
            sn -= (f(N2 - j) + f(N2 - j - 1))*0.5
        return sn, Sn
    elif n > N2:
        resets()
        for j in range(n - N2):
            Sn += sn + f(N2 + j)*0.5
            sn += (f(N2 + j) + f(N2 + j + 1))*0.5
        return sn, Sn

def getsr(n):
    global Sn, sn
    if n == N + 1:
        resets()
        for j in range(n - N2 - 1):
            if j != 0:
                sn += (f(N2 + j - 1) + f(N2 + j))*0.5
            Sn += sn + f(N2 + j)*0.5

    sn += (f(n - 2) + f(n - 1))*0.5
    Sn += sn + f(n - 1)*0.5
    return sn, Sn

def getssr(n):
    global sn
    return sn + (f(n - 1) + f(n))*0.5

maxa = 1
while maxa > 0.00000000001:
    maxa = 0
    for n in range(N + 1):
        if n != N2:
            s, S = getss(n)
            aold = f(n)
            #correct starting value
            sum3r = np.array([0,0,0], dtype=np.longdouble)
            sum3v = np.array([0,0,0], dtype=np.longdouble)
            for k in range(N + 1):
                a3 = f(k)
                sum3r += a3*a[n][k]
                sum3v += a3*b[n][k]
            r[n] = (S + sum3r)*dt**2
            v[n] = (s + sum3v)*dt
            #check convergence of accelerations
            anew = f(n)
            magdif = mag(aold - anew)
            if magdif > maxa:
                maxa = magdif


start = time()
#Commencing PEC cycle:
n = N
t = N2*dt

corrsum = np.empty([2, 3], dtype=np.longdouble)
with tqdm(total=(stepn - N2 - 1)) as pbar:
    while t < TP*nrev:        #T is defined in general_definitions
        #Predict:
        s, S = getsr(n + 1)
        psum = np.array([0,0,0], dtype=np.longdouble)
        psumv = np.array([0,0,0], dtype=np.longdouble)
        for k in range(N + 1):
            ap = f(n-N+k)
            psum += ap*a[N + 1][k]
            psumv+= ap*b[N + 1][k]
        r[n + 1] = (psum + S)*dt**2
        v[n + 1] = (psumv + f(n)/2 + psumv)*dt
        n += 1
        corrsum.fill(0)
        #Evaluate-Correct:
        for k in range(N):
            ac = f(n + k - N)
            corrsum[0] += ac*a[N][k]
            corrsum[1] += ac*b[N][k]

        for _ in range(200):
            max = 0
            rold = r[n]
            r[n] = (f(n)*a[N][N] + corrsum[0] + S)*dt**2
            v[n] = (f(n)*b[N][N] + corrsum[1] + s)*dt
            diff = mag(rold - r[n])
            if diff > max:
                max = diff
            if max < 0.0000000001:
                break
        t += dt
        pbar.update(1)



print("Time elapsed {}s".format(time() - start))
print(r[-1])
print("Gauss-Jackson with step-size {}s after {} revolutions has an imprecision of {}%, {}%".format(dt, nrev, mag(r[-1] - r0)/mag(r0), mag(v[-1] - v0)/mag(v0)))



# rb = r[0:17:2]
# vb = v[0:17:2]

# dt = dt*2
# stepn = int(TP*nrev/dt + 2)


# r = np.empty([stepn + N2,3], dtype=np.longdouble)
# r[N2] = r0
# v = np.empty([stepn + N2,3], dtype=np.longdouble)
# v[N2] = v0

# r[0:9] = rb
# v[0:9] = vb


# start = time()
# #Commencing PEC cycle:
# n = N
# t = N2*dt

# corrsum = np.empty([2, 3], dtype=np.longdouble)
# with tqdm(total=(stepn - N2 - 1)) as pbar:
#     while t < TP*nrev:        #T is defined in general_definitions
#         #Predict:
#         s, S = getsr(n + 1)
#         psum = np.array([0,0,0], dtype=np.longdouble)
#         psumv = np.array([0,0,0], dtype=np.longdouble)
#         for k in range(N + 1):
#             ap = f(n-N+k)
#             psum += ap*a[N + 1][k]
#             psumv+= ap*b[N + 1][k]
#         r[n + 1] = (psum + S)*dt**2
#         v[n + 1] = (psumv + f(n)/2 + psumv)*dt
#         n += 1
#         corrsum.fill(0)
#         #Evaluate-Correct:
#         for k in range(N):
#             ac = f(n + k - N)
#             corrsum[0] += ac*a[N][k]
#             corrsum[1] += ac*b[N][k]

#         for _ in range(200):
#             max = 0
#             rold = r[n]
#             r[n] = (f(n)*a[N][N] + corrsum[0] + S)*dt**2
#             v[n] = (f(n)*b[N][N] + corrsum[1] + s)*dt
#             diff = mag(rold - r[n])
#             if diff > max:
#                 max = diff
#             if max < 0.0000000001:
#                 break
#         t += dt
#         pbar.update(1)



# print("Time elapsed {}s".format(time() - start))
# print("Gauss-Jackson with step-size {}s after {} revolutions has an imprecision of {}%, {}%".format(dt, nrev, mag(r[-1] - r0)/mag(r0), mag(v[-1] - v0)/mag(v0)))