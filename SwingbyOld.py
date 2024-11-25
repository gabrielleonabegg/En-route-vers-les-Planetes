import numpy as np
from helpers1 import G, mag, norm
from general_definitions import dt, N2, masses, rad_of_closest_approach, earthUnitRotationVector, tick_rate, unitRotationVector, names, pEpoch
from kepler import C, S
from math import factorial, cos


import warnings
from sys import exit
import datetime
import tqdm
from tqdm import trange


def unsigned_angle(a:np.ndarray, b:np.ndarray):
    return np.arccos(np.dot(a,b)/(mag(a)*mag(b)))

def unsignedAngle(array1:np.ndarray, array2:np.ndarray):    #takes two arrays of vectors as inputs, speeds up work with large datasets as we don't need a for loop (about 60 times faster for large arrays, but since both operations are rather fast, we don't actually care)
    if array1.shape != array2.shape:
        print("Arrays are not of the same shape")
        return None
    if array1.shape[-1] != 3:
        print("Warning: 3-dimentional vectors are expected")
    dot = np.sum(array1*array2, axis=-1)
    mag = np.sqrt(np.sum(array1**2, -1)*np.sum(array2**2, -1))
    return np.arccos(dot/mag)


def signed_angle(a:np.ndarray, b:np.ndarray, N:np.ndarray):      #https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
    alpha = np.arctan2(np.dot(np.cross(a,b), N), np.dot(a, b))
    if alpha > 0:
        return alpha
    else:
        return 2*np.pi + alpha

def derC(z):#Checked
    fact = 4
    k = 1
    s = 0
    deltas = 100
    while abs(deltas) > 1e-7:
        # print("{}*z^{}/{}!".format((-1)**k * k, k - 1, fact))
        deltas = (-1)**k*k/factorial(fact)*z**(k - 1)
        s += deltas
        k +=1
        fact+=2
    return s



def derS(z):#Checked
    fact = 5
    k = 1
    s = 0
    deltas = 100
    while abs(deltas) > 1e-7:
        # print("{}*z^{}/{}!".format((-1)**k*k, (k - 1), fact))
        deltas = (-1)**k*k/factorial(fact)*z**(k - 1)
        s += deltas
        k +=1
        fact+=2
    return s

N2order = N2    #later used for the normal vector
def timeToIndex(t):
    return abs(int(round(t/dt)) + N2order)

def hyperbolicTOF(mu, a, e, true_anomaly):#BUG CHECK
    # print("a: {}".format((-a)**3))
    F = np.arccosh((e + np.cos(true_anomaly))/(1 + e*np.cos(true_anomaly)))
    # if true_anomaly > np.pi and true_anomaly < 2*np.pi:
    #     F = -F
    return abs(np.sqrt((-a)**3/mu)*(e*np.sinh(F) - F))

AHHcount = 0
def porkchop(t1, t2, p1_index, p2_index, r1v=None, r2v=None):       #assuming time from ephemeris epoch; CHECKED
    global AHHcount
    time_1_index = timeToIndex(t1)
    time_2_index = timeToIndex(t2)
    case0 = False
    if r1v is None or r2v is None:
        r1v  = rplanets[time_1_index, p1_index] - rplanets[time_1_index, 0]
        r2v  = rplanets[time_2_index, p2_index] - rplanets[time_2_index, 0]
        case0 = True
    v1v = vplanets[time_1_index, p1_index] - vplanets[time_1_index, 0]
    v2v = vplanets[time_2_index, p2_index] - vplanets[time_2_index, 0]
    mu = G*masses[0]
    t = t2 - t1
    n = np.cross(r1v, r2v)
    h = np.cross(r1v, v1v)
    dv = unsigned_angle(r1v, r2v)
    if unsigned_angle(n, h) > np.pi/2:      #Genaue überlegungen: siehe arbeitsjournal
        DM = -1
        dv = 2*np.pi - dv
    else:
        DM = 1
        
    sqrtmu = np.sqrt(mu)
    if abs(dv - np.pi) < 1e-3:
        tqdm.tqdm.write("Collinear state vectors: the plane is not unequely defined")
        return None
    if abs(dv - 2*np.pi) < 1e-3 or dv < 1e-3:
        tqdm.tqdm.write("Collinear state vectors that point in the same direction; isn't optimal anyways")
        return None
        #CHANGE
    r1 = mag(r1v)
    r2 = mag(r2v)
    A = DM*(np.sqrt(r1*r2)*np.sin(dv))/(np.sqrt(1 - np.cos(dv)))
    zn = dv**2
    convergence = False
    for _ in range(50):
        Sn = S(zn)
        Cn = C(zn)
        y = r1 + r2 - A*(1 - zn*Sn)/np.sqrt(Cn)
        if y < 0:
            AHHcount += 1
            return None
        xn = np.sqrt(y/Cn)
        tn = xn**3*Sn/sqrtmu + A*np.sqrt(y)/sqrtmu
        if t < 1 or t < 1e6:
            if abs(t - tn) < 1e-4:
                convergence = True
                break
        elif abs((t - tn)/t) < 1e-4:      #These values no longer correspond to one another (new and old edition) WHY ARE THESE VALUES NORMALIZED
            convergence = True
            break
        if abs(zn) < 0.5:
            dCdz = derC(zn)
            dSdz = derS(zn)
        else:
            dCdz = 1/(2*zn)*(1 - zn*Sn - 2*Cn)
            dSdz = 1/(2*zn)*(Cn - 3*Sn)

        dtdz = xn**3/sqrtmu*(dSdz - (3*Sn*dCdz)/(2*Cn)) + A/(8*sqrtmu)*((3*Sn*np.sqrt(y))/Cn + A/xn)
        zn = zn + (t - tn)/dtdz
    if not convergence:
        tqdm.tqdm.write("The method hasn't converged, returning approximate values")
    f = 1 - y/r1
    g = A*np.sqrt(y/mu)
    dg = 1 - y/r2
    v1 = (r2v - f*r1v)/g
    v2 = (dg*r2v - r1v)/g
    vinf_departure = v1 - v1v
    vinf_arrival = v2 - v2v
    if case0:
        C3 = mag(vinf_departure)**2
        return C3, vinf_departure, vinf_arrival
    else:
        return vinf_departure, vinf_arrival


#Load ephemerides of all Planets:
with open("r.npy", "rb") as file:
    rplanets = np.load(file)
with open("v.npy", "rb") as file:
    vplanets = np.load(file)
now = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
# epoch = datetime.datetime.fromtimestamp(os.path.getmtime('r.npy')).replace(hour=0, minute=0, second=0, microsecond=0) #NOTE: outdated
epoch = pEpoch()
deltat = (now - epoch).total_seconds()       #Time in seconds since ephemeris "epoch"

transfer_time = np.array([[0, 0], [0.4, 1], [2, 3], [3, 7], [10, 17], [10, 31]], dtype=np.longdouble)*31536000        #transfer time from earth [min, max] in years; for mars a range of 1 year is used (which is more than a Hohmann transfer) and 0.4 is slightly lower than the record

periapsisEarthOrbit = 185000 + 6378137      #According to Vallado: 



mode0 = False
mode1 = False
mode2 = False


check = True
while check:
    print("What is your target ?\n6)  Saturn\n7)  Uranus\n8)  Neptune")
    x = input()
    try:
        x = int(x)
        if x < 6 or x > 8:
            print("Invalid index {} cannot be chosen".format(x))
        else:
            check = False
    except ValueError:
        print("Please provide an integer")


departure1 = 3#Earth
arrival1 = 5#Jupiter
departure2 = 5
arrival2 = x#Arrival
range_of_analysis = 12*31536000         #NOTE:Arbitrary choice, to be verified
mu1 = G*masses[departure1]
mu2 = G*masses[arrival1]
mu3 = G*masses[arrival2]

maxRadiusAtArrival = np.min(mag(rplanets[::100, arrival2] - rplanets[::100, 0]))*(masses[arrival2]/masses[0])**(2/5)          #perihelion is used; I only use every 100th point. With a step-size of 1000s this is around 1.2 days, still totally acceptable while a lot faster
print("Input the arrival orbit's radius (min. : {})".format(rad_of_closest_approach[arrival2]))
while True:
    arrivalOrbitRadius = input()
    try:
        arrivalOrbitRadius = np.longdouble(arrivalOrbitRadius)
        if arrivalOrbitRadius > maxRadiusAtArrival:
            print("Specified radius greater than the SOI of {}".format(names[arrival2]))
        elif arrivalOrbitRadius < rad_of_closest_approach[arrival2]:
            print("Radius is less than the specified lower limit. Do you want to use this radius nonetheless? (y/n)")
            if input().capitalize() == "Y":
                break
        else:
            break
    except ValueError:
        print("Please input a number")

transfer_time = np.array([[np.nan,np.nan],[0.1, 3], [0.2, 0.7], [0, 0], [0.4, 1], [2, 3], [3, 7], [10, 17], [10, 31]], dtype=np.longdouble)*31536000        #transfer time from earth [min, max] in years; for mars a range of 1 year is used (which is more than a Hohmann transfer) and 0.4 is slightly lower than the record; rough estimates are used for mercury and venus: In order to establish an orbit around mercury, one would have to make multiple venus flybys; for venus: lower limit roughly four months upper limit: more than the hohmann transfer
transfer_range1 =  [transfer_time[arrival1, 0] - transfer_time[departure1, 1], transfer_time[arrival1, 1] - transfer_time[departure1, 0]]
transfer_range2 =  [transfer_time[arrival2, 0] - transfer_time[departure2, 1], transfer_time[arrival2, 1] - transfer_time[departure2, 0]]
search_ranget1 = transfer_range1[1] - transfer_range1[0]
search_ranget2 = transfer_range2[1] - transfer_range2[0]

i = 0
sd1 = int(range_of_analysis/tick_rate[departure1])
sd2 = int(search_ranget1/tick_rate[arrival1]) + 1
pchp1 = np.empty([sd1, sd2, 5], dtype=np.longdouble)
jmax = 0
for t1 in trange(int(deltat), int(deltat + range_of_analysis), int(tick_rate[departure1])):
    j = 0
    for t2 in range(int(t1) + int(transfer_range1[0]), int(t1) + int(transfer_range1[1]), int(tick_rate[arrival1])):
        tmp = porkchop(t1, t2, departure1, arrival1)
        if tmp is not None:
            C3, vinf_departure, vinf_arrival = tmp
            vinf_departure = mag(vinf_departure)
            pchp1[i, j] = C3, vinf_departure, *vinf_arrival
        else:
            pchp1[i, j] = np.nan, np.nan, np.nan, np.nan, np.nan
        j += 1
    if i == 0:
        jmax = j - 1
    i += 1



sd1 = int((range_of_analysis + transfer_range1[1] - transfer_range1[0])/tick_rate[departure2]) + 1
sd2 = int(search_ranget2/tick_rate[arrival2]) + 1
pchp2 = np.empty([sd1, sd2, 4])
i = 0
for t1 in trange(int(deltat + transfer_range1[0]), int(deltat + transfer_range1[1] + range_of_analysis), int(tick_rate[departure2])):
    j = 0
    for t2 in range(t1 + int(transfer_range2[0]), int(t1 + transfer_range2[1]), int(tick_rate[arrival2])):
        tmp = porkchop(t1, t2, departure2, arrival2)
        if tmp is not None:
            _, vinf_departure, vinf_arrival = tmp
            vinf_arrival = mag(vinf_arrival)
            pchp2[i, j] = *vinf_departure, vinf_arrival
        else:
            pchp2[i, j] = np.nan, np.nan, np.nan, np.nan
        j += 1
    i += 1

tqdm.tqdm.write("AHHHH*({})".format(AHHcount))

def swingbyTimeIndices(t):#Checked; returns the all indices that indicate an arrival time at target of t
    start = int(round((t - (transfer_range1[0] + jmax*tick_rate[arrival1]) - deltat)/tick_rate[departure1]))
    stop = int(round((t - transfer_range1[0] - deltat)/tick_rate[departure1])) + 1
    i = np.arange(start if start > 0 else 0, stop if stop < int(range_of_analysis/tick_rate[departure1]) else int(range_of_analysis/tick_rate[departure1]))        # + 1 because the last element should be inclusive; the starting index for i cannot be lower than 0 and the last index mustn't exceed the highest possible index for i
    pchp1Indices = (i, np.rint((t - i*tick_rate[departure1] - deltat - transfer_range1[0])/tick_rate[arrival1]).astype(np.int64),)
    return pchp1Indices

k = 0
compare = np.empty([sd1], dtype=np.dtype([('dvinf', np.longdouble), ('indices', np.int64, (2, 2))]))
compare[:]['dvinf'] = np.nan
for swingby_time in trange(int(deltat + transfer_range1[0]), int(deltat + transfer_range1[1] + range_of_analysis), int(tick_rate[departure2])):     #for every possible swingby date, select the departure and arrival dates that are best
    pchp1Indices = swingbyTimeIndices(swingby_time)
    if not np.size(pchp1Indices[0]):
        break
    jarray1 = pchp1[pchp1Indices]
    jarray2 = pchp2[k]        
    min = np.inf
    m = None    #add safety if the two conditions below should be unfulfilled
    for i in range(len(jarray1)):
        for j in range(len(jarray2)):
            if mu2/mag(jarray1[i, 2:5])**2 * (1/cos((np.pi - unsignedAngle(jarray1[i, 2:5], jarray2[j, 0:3]))/2) - 1) > rad_of_closest_approach[arrival1]:  #Check if physically possible without crashing
                magdiff = abs(mag(jarray1[i, 2:5]) - mag(jarray2[j, 0:3]))
                if magdiff < 1:  #minimise the difference in magnitude of the ingoing and outgoint vinf vectors at the swingby planet
                    if jarray1[i, 1] < min:
                        min = jarray1[i, 1]
                        m = i, j
    if m is not None:
        i, j = m
        compare[k]['indices'] = (pchp1Indices[0][i], pchp1Indices[1][i]), (k, j)
        compare[k]['dvinf'] = abs(mag(jarray1[i, 2:5]) - mag(jarray2[j, 0:3]))
    else:
        compare[k]['indices'] = (0, 0), (0, 0)
        compare[k]['dvinf'] = np.nan
    k += 1

#first we will choose those with low velocity difference @swingby
indices = np.where(compare['dvinf'] < 1)
# print(indices)
compare = compare[indices]
# print(compare)
#we have now chosen those favorable at swingby. We also want to minimize Δv at departure and arrival. We will do this as follows:
#first: make indices for pchp1&2
pchp1OptimizedIndices = tuple([compare['indices'][:,0, 0], compare['indices'][:,0, 1]])
pchp2OptimizedIndices = tuple([compare['indices'][:,1, 0], compare['indices'][:,1, 1]])

# print(pchp1[pchp1OptimizedIndices][:, 1])
# print(pchp2[pchp2OptimizedIndices])

cvdep = np.sqrt(mu1/periapsisEarthOrbit)        #circular speed at departure
cvarr = np.sqrt(mu3/arrivalOrbitRadius)

#From the Energy equation follows that at distance r = infinity; E=vinf^2/2 and at periapsis = vp^2 / 2 - mu/rp
            #1-Dimentional array, np.nanargmin is ok!
minindex = np.nanargmin(np.sqrt(pchp1[pchp1OptimizedIndices][:, 1]**2 + 2*mu1/periapsisEarthOrbit) - np.sqrt(mu1/periapsisEarthOrbit) + np.sqrt(pchp2[pchp2OptimizedIndices][:, 3]**2 + 2*mu3/arrivalOrbitRadius) - np.sqrt(mu3/arrivalOrbitRadius))


pchp1Indices = pchp1OptimizedIndices[0][minindex], pchp1OptimizedIndices[1][minindex]
pchp2Indices = pchp2OptimizedIndices[0][minindex], pchp2OptimizedIndices[1][minindex]

#NOTE: Add tests here
t1 = pchp1Indices[0]*tick_rate[departure1] + deltat
t2 = pchp2Indices[0]*tick_rate[departure2] + transfer_range1[0] + deltat
t3 = t2 + transfer_range2[0] + pchp2Indices[1]*tick_rate[arrival2]


# print(t1)
# print(t2)
# print(t3)
departureIndex = timeToIndex(t1)
flybyIndex = timeToIndex(t2)
arrivalIndex = timeToIndex(t3)

_, vinf_dep1, vinf_arr1 = porkchop(t1, t2, departure1, arrival1)
_, vinf_dep2, vinf_arr2 = porkchop(t2, t3, departure2, arrival2)

# print(mag(vinf_arr1) - mag(vinf_dep2))
# print(mag(vinf_dep1))
# print(mag(vinf_arr2))

# rdep = rplanets[departureIndex, departure1] - rplanets[departureIndex, 0]
# rarr = rplanets[flybyIndex, arrival1] - rplanets[flybyIndex, 0]
# vdep = vplanets[departureIndex, departure1] - vplanets[departureIndex, 0] + vinf_dep1
# varr = vplanets[flybyIndex, arrival1] - vplanets[flybyIndex, 0] + vinf_arr1

# muS = G*masses[0]
# evect = (mag(vdep)**2/muS - 1/mag(rdep))*rdep - np.dot(rdep, vdep)/muS*vdep
# hvect = np.cross(rdep, vdep)
# true_anomaly_dep = signed_angle(evect, rdep, norm(hvect))
# true_anomaly_arr = signed_angle(evect, rarr, norm(hvect))
# print("T1: Value ought to be near 0: {}".format(mag(evect - ((mag(varr)**2/muS - 1/mag(rarr))*rarr - np.dot(rarr, varr)/muS*varr))))    #Do the same thing with arrival and departure to verify that they describe the same thing
# print("T2: Value ought to be near 0: {}".format(mag(hvect - np.cross(rarr, varr))))
# p = mag(hvect)**2/muS
# e = mag(evect)
# print(e)
# P = norm(evect)
# Q = norm(np.cross(hvect, evect))
# def plotter(true_anomaly):
#     return p/(1 + e*np.cos(true_anomaly))*np.cos(true_anomaly)*P + p/(1 + e*np.cos(true_anomaly))*np.sin(true_anomaly)*Q
# trajectory = np.empty([10000, 3], dtype=np.longdouble)
# i = 0
# bool1 = True
# for true_anomaly in np.arange(true_anomaly_dep, true_anomaly_arr, (true_anomaly_arr - true_anomaly_dep)/10000):
#     if bool1:
#         print(true_anomaly)
#         bool1 = False
#     trajectory[i] = plotter(true_anomaly)
#     i += 1
# print(true_anomaly_arr, "2.549362280593937506")
# print(rdep, plotter(1.0563201566314143629))
# print(rarr, plotter(2.549362280593937506))
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# ax = plt.axes(projection="3d")
# ax.scatter(*rdep)
# ax.scatter(*rarr)
# ax.plot(*trajectory.T)
# plt.show()


periapsisEarthOrbit = 185000 + 6378137      #According to Vallado: 

departureSOI = mag(rplanets[departureIndex, departure1] - rplanets[departureIndex, 0])*(masses[departure1]/masses[0])**(2/5)
swingbySOI = mag(rplanets[flybyIndex, departure2] - rplanets[flybyIndex, 0])*(masses[departure2]/masses[0])**(2/5)
arrivalSOI = mag(rplanets[arrivalIndex, arrival2] - rplanets[arrivalIndex, 0])*(masses[arrival2]/masses[0])**(2/5)


#NOTE : To be verified
rotationVectorEarth = earthUnitRotationVector(t1)       #Not going to change a lot
rotationVectorTarget = unitRotationVector(t3, x)
#SOI of earth
s = - np.sum(vinf_dep1*rotationVectorEarth)/np.sum(vinf_dep1**2)
N = norm(rotationVectorEarth + s*vinf_dep1)
# print("Important new test 1")
# print(np.dot(N, vinf_dep1))
e = mag(vinf_dep1)**2*periapsisEarthOrbit/(mu1) + 1        #See notes made in Vallado
p = (mag(vinf_dep1)**2*periapsisEarthOrbit**2/(mu1) + 2*periapsisEarthOrbit)        #See notes made in Vallado
true_anomaly = abs(np.arccos(-1/e))     #@infinity
# print(true_anomaly)
P = np.array([1/(1 + e**2 + 2 * e * np.cos(true_anomaly)) * np.sqrt(p/mu1) * (N[2] * vinf_dep1[1] * (e + np.cos(true_anomaly)) - N[1] * vinf_dep1[2] * (e + np.cos(true_anomaly)) - 2 * e * vinf_dep1[0] / np.tan(true_anomaly) - vinf_dep1[0] / np.sin(true_anomaly) - e**2 * vinf_dep1[0] / np.sin(true_anomaly) + N[1]**2 * vinf_dep1[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + N[2]**2 * vinf_dep1[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N[0] * N[1] * vinf_dep1[1] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N[0] * N[2] * vinf_dep1[2] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly)), -((np.sqrt(p/mu1) * (2 * N[2] * vinf_dep1[0] * (e + np.cos(true_anomaly)) - 2 * N[0] * vinf_dep1[2] * (e + np.cos(true_anomaly)) + vinf_dep1[1] / np.sin(true_anomaly) + 2 * N[0] * N[1] * vinf_dep1[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N[1]**2 * vinf_dep1[1] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N[1] * N[2] * vinf_dep1[2] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - vinf_dep1[1] * np.cos(2 * true_anomaly) / np.sin(true_anomaly)))/(2 * (1 + e**2 + 2 * e * np.cos(true_anomaly)))), -((np.sqrt(p/mu1) * (vinf_dep1[2] - N[1] * vinf_dep1[0] * (e + np.cos(true_anomaly)) / np.sin(true_anomaly) + N[0] * vinf_dep1[1] * (e + np.cos(true_anomaly)) / np.sin(true_anomaly) + N[0] * N[2] * vinf_dep1[0] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2 + N[1] * N[2] * vinf_dep1[1] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2 + N[2]**2 * vinf_dep1[2] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2) * np.sin(true_anomaly))/(1 + e**2 + 2 * e * np.cos(true_anomaly)))], dtype=np.longdouble)
# print("TEST 1: VALUE OUGHT TO BE NEAR 0: {}".format(N.dot(P)))
# print("TEST 2: Value ought to be near 1: {}".format(mag(P)))
Q = np.cross(N, P)
vtest = np.sqrt(mu1/p)*(-np.sin(true_anomaly)*P + (e + np.cos(true_anomaly))*Q)
# print("Important new test 2")
# print(mag(vinf_dep1 - vtest))
true_anomalySOI = np.arccos((p/departureSOI - 1)/e)
departureVector1 = (p - departureSOI)/e*P + departureSOI*np.sin(true_anomalySOI)*Q + rplanets[departureIndex, departure1] - rplanets[departureIndex, 0]
#Swingby SOI
N = norm(np.cross(vinf_arr1, vinf_dep2))
P = norm(norm(vinf_arr1) - norm(vinf_dep2))
Q = norm(norm(vinf_arr1) + norm(vinf_dep2))
# print("Test1: Value ought to be near 0: {}".format(N.dot(P)))
# print("Test2: Value ought to be near 0: {}".format(mag(np.cross(P, Q) - N)))
turning_angle = unsigned_angle(vinf_arr1, vinf_dep2)
rp = mu2/mag(vinf_arr1)**2*(1/cos((np.pi - turning_angle)/2) - 1)
e = 1/np.sin(turning_angle/2)
p = mag(vinf_arr1)**2 * rp**2 /mu2 + 2*rp
true_anomalySOIout = np.arccos((p/swingbySOI - 1)/e)
true_anomalySOIin = 2*np.pi - true_anomalySOIout
swingbyTOF2 = hyperbolicTOF(mu2,  p/(1 - e**2), e, true_anomalySOIout)
TimeIn = t2 - swingbyTOF2
TimeInIndex = timeToIndex(TimeIn)
TimeOut = t2 + swingbyTOF2
TimeOutIndex = timeToIndex(TimeOut)
rvSOIin = swingbySOI*cos(true_anomalySOIin)*P + swingbySOI*np.sin(true_anomalySOIin)*Q
rvSOIout = swingbySOI*cos(true_anomalySOIout)*P + swingbySOI*np.sin(true_anomalySOIout)*Q
arrivalVector1 = rvSOIin + rplanets[TimeInIndex, arrival1] - rplanets[TimeInIndex, 0]
departureVector2 = rvSOIout + rplanets[TimeOutIndex, departure2] - rplanets[TimeOutIndex, 0]
#Define arrival SOI:
s = - np.sum(vinf_arr2*rotationVectorTarget)/np.sum(vinf_arr2**2)
N = (rotationVectorTarget + s*vinf_arr2)/mag(rotationVectorTarget + s*vinf_arr2)
# print("Test: Value ought to be near 0: {}".format(N.dot(vinf_arr2)))
e = mag(vinf_arr2)**2*arrivalOrbitRadius/(mu3) + 1
p = (mag(vinf_arr2)**2*arrivalOrbitRadius**2/(mu3) + 2*arrivalOrbitRadius)
true_anomaly = abs(np.arccos(-1/e))
P = np.array([1/(1 + e**2 + 2 * e * np.cos(true_anomaly)) * np.sqrt(p/mu3) * (N[2] * vinf_arr2[1] * (e + np.cos(true_anomaly)) - N[1] * vinf_arr2[2] * (e + np.cos(true_anomaly)) - 2 * e * vinf_arr2[0] / np.tan(true_anomaly) - vinf_arr2[0] / np.sin(true_anomaly) - e**2 * vinf_arr2[0] / np.sin(true_anomaly) + N[1]**2 * vinf_arr2[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + N[2]**2 * vinf_arr2[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N[0] * N[1] * vinf_arr2[1] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N[0] * N[2] * vinf_arr2[2] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly)), -((np.sqrt(p/mu3) * (2 * N[2] * vinf_arr2[0] * (e + np.cos(true_anomaly)) - 2 * N[0] * vinf_arr2[2] * (e + np.cos(true_anomaly)) + vinf_arr2[1] / np.sin(true_anomaly) + 2 * N[0] * N[1] * vinf_arr2[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N[1]**2 * vinf_arr2[1] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N[1] * N[2] * vinf_arr2[2] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - vinf_arr2[1] * np.cos(2 * true_anomaly) / np.sin(true_anomaly)))/(2 * (1 + e**2 + 2 * e * np.cos(true_anomaly)))), -((np.sqrt(p/mu3) * (vinf_arr2[2] - N[1] * vinf_arr2[0] * (e + np.cos(true_anomaly)) / np.sin(true_anomaly) + N[0] * vinf_arr2[1] * (e + np.cos(true_anomaly)) / np.sin(true_anomaly) + N[0] * N[2] * vinf_arr2[0] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2 + N[1] * N[2] * vinf_arr2[1] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2 + N[2]**2 * vinf_arr2[2] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2) * np.sin(true_anomaly))/(1 + e**2 + 2 * e * np.cos(true_anomaly)))], dtype=np.longdouble)
Q = np.cross(N, P)
true_anomalySOI = np.arccos((p/arrivalSOI - 1)/e)
arrivalVector2 = (p - arrivalSOI)/e*P + arrivalSOI*np.sin(true_anomalySOI)*Q + rplanets[arrivalIndex, arrival2] - rplanets[arrivalIndex, 0]
while True:
    vinf_dep1, vinf_arr1 = porkchop(t1, TimeIn, departure1, arrival1, departureVector1, arrivalVector1)
    vinf_dep2, vinf_arr2 = porkchop(TimeOut, t3, departure2, arrival2, departureVector2, arrivalVector2)
    s = - np.sum(vinf_dep1*rotationVectorEarth)/np.sum(vinf_dep1**2)
    N1 = norm(rotationVectorEarth + s*vinf_dep1)
    e1 = mag(vinf_dep1)**2*periapsisEarthOrbit/(mu1) + 1        #See notes made in Vallado
    p1 = (mag(vinf_dep1)**2*periapsisEarthOrbit**2/(mu1) + 2*periapsisEarthOrbit)        #See notes made in Vallado
    true_anomaly = abs(np.arccos(-1/e1))     #@infinity
    P1 = np.array([1/(1 + e1**2 + 2 * e1 * np.cos(true_anomaly)) * np.sqrt(p1/mu1) * (N1[2] * vinf_dep1[1] * (e1 + np.cos(true_anomaly)) - N1[1] * vinf_dep1[2] * (e1 + np.cos(true_anomaly)) - 2 * e1 * vinf_dep1[0] / np.tan(true_anomaly) - vinf_dep1[0] / np.sin(true_anomaly) - e1**2 * vinf_dep1[0] / np.sin(true_anomaly) + N1[1]**2 * vinf_dep1[0] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + N1[2]**2 * vinf_dep1[0] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N1[0] * N1[1] * vinf_dep1[1] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N1[0] * N1[2] * vinf_dep1[2] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly)), -((np.sqrt(p1/mu1) * (2 * N1[2] * vinf_dep1[0] * (e1 + np.cos(true_anomaly)) - 2 * N1[0] * vinf_dep1[2] * (e1 + np.cos(true_anomaly)) + vinf_dep1[1] / np.sin(true_anomaly) + 2 * N1[0] * N1[1] * vinf_dep1[0] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N1[1]**2 * vinf_dep1[1] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N1[1] * N1[2] * vinf_dep1[2] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - vinf_dep1[1] * np.cos(2 * true_anomaly) / np.sin(true_anomaly)))/(2 * (1 + e1**2 + 2 * e1 * np.cos(true_anomaly)))), -((np.sqrt(p1/mu1) * (vinf_dep1[2] - N1[1] * vinf_dep1[0] * (e1 + np.cos(true_anomaly)) / np.sin(true_anomaly) + N1[0] * vinf_dep1[1] * (e1 + np.cos(true_anomaly)) / np.sin(true_anomaly) + N1[0] * N1[2] * vinf_dep1[0] * (1/np.tan(true_anomaly) + e1 / np.sin(true_anomaly))**2 + N1[1] * N1[2] * vinf_dep1[1] * (1/np.tan(true_anomaly) + e1 / np.sin(true_anomaly))**2 + N1[2]**2 * vinf_dep1[2] * (1/np.tan(true_anomaly) + e1 / np.sin(true_anomaly))**2) * np.sin(true_anomaly))/(1 + e1**2 + 2 * e1 * np.cos(true_anomaly)))], dtype=np.longdouble)
    # print("TEST 1: VALUE OUGHT TO BE NEAR 0: {}".format(N1.dot(P1)))
    # print("TEST 2: Value ought to be near 1: {}".format(mag(P1)))
    Q1 = np.cross(N1, P1)
    true_anomalySOI1 = np.arccos((p1/departureSOI - 1)/e1)
    diff0 = departureVector1
    departureVector1 = (p1 - departureSOI)/e1*P1 + departureSOI*np.sin(true_anomalySOI1)*Q1 + rplanets[departureIndex, departure1] - rplanets[departureIndex, 0]
    diff0 = mag(departureVector1 - diff0)
    #Swingby SOI
    N2 = norm(np.cross(vinf_arr1, vinf_dep2))
    P2 = norm(norm(vinf_arr1) - norm(vinf_dep2))
    Q2 = norm(norm(vinf_arr1) + norm(vinf_dep2))
    # print("Test1: Value ought to be near 0: {}".format(N2.dot(P2)))
    # print("Test2: Value ought to be near 0: {}".format(mag(np.cross(P2, Q2) - N2)))
    turning_angle = unsigned_angle(vinf_arr1, vinf_dep2)
    rp = mu2/mag(vinf_arr1)**2*(1/cos((np.pi - turning_angle)/2) - 1)
    e2 = 1/np.sin(turning_angle/2)
    p2 = mag(vinf_arr1)**2 * rp**2 /mu2 + 2*rp
    true_anomalySOIout = np.arccos((p2/swingbySOI - 1)/e2)
    true_anomalySOIin = 2*np.pi - true_anomalySOIout

    swingbyTOF2 = hyperbolicTOF(mu2,  p2/(1 - e2**2), e2, true_anomalySOIout)
    TimeIn = t2 - swingbyTOF2
    TimeInIndex = timeToIndex(TimeIn)
    TimeOut = t2 + swingbyTOF2
    TimeOutIndex = timeToIndex(TimeOut)
    rvSOIin = swingbySOI*cos(true_anomalySOIin)*P2 + swingbySOI*np.sin(true_anomalySOIin)*Q2
    rvSOIout = swingbySOI*cos(true_anomalySOIout)*P2 + swingbySOI*np.sin(true_anomalySOIout)*Q2
    diff1 = arrivalVector1
    arrivalVector1 = rvSOIin + rplanets[TimeInIndex, arrival1] - rplanets[TimeInIndex, 0]
    diff1 = mag(arrivalVector1 - diff1)
    diff2 = departureVector2
    departureVector2 = rvSOIout + rplanets[TimeOutIndex, departure2] - rplanets[TimeOutIndex, 0]
    diff2 = mag(departureVector2 - diff2)
    #Define arrival SOI:
    s = - np.sum(vinf_arr2*rotationVectorTarget)/np.sum(vinf_arr2**2)
    N3 = (rotationVectorTarget + s*vinf_arr2)/mag(rotationVectorTarget + s*vinf_arr2)
    # print("Test: Value ought to be near 0: {}".format(N3.dot(vinf_arr2)))
    e3 = mag(vinf_arr2)**2*arrivalOrbitRadius/(mu3) + 1
    p3 = (mag(vinf_arr2)**2*arrivalOrbitRadius**2/(mu3) + 2*arrivalOrbitRadius)
    true_anomaly = abs(np.arccos(-1/e3))#see p 17 notes
    P3 = np.array([1/(1 + e3**2 + 2 * e3 * np.cos(true_anomaly)) * np.sqrt(p3/mu3) * (N3[2] * vinf_arr2[1] * (e3 + np.cos(true_anomaly)) - N3[1] * vinf_arr2[2] * (e3 + np.cos(true_anomaly)) - 2 * e3 * vinf_arr2[0] / np.tan(true_anomaly) - vinf_arr2[0] / np.sin(true_anomaly) - e3**2 * vinf_arr2[0] / np.sin(true_anomaly) + N3[1]**2 * vinf_arr2[0] * (e3 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + N3[2]**2 * vinf_arr2[0] * (e3 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N3[0] * N3[1] * vinf_arr2[1] * (e3 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N3[0] * N3[2] * vinf_arr2[2] * (e3 + np.cos(true_anomaly))**2 / np.sin(true_anomaly)), -((np.sqrt(p3/mu3) * (2 * N3[2] * vinf_arr2[0] * (e3 + np.cos(true_anomaly)) - 2 * N3[0] * vinf_arr2[2] * (e3 + np.cos(true_anomaly)) + vinf_arr2[1] / np.sin(true_anomaly) + 2 * N3[0] * N3[1] * vinf_arr2[0] * (e3 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N3[1]**2 * vinf_arr2[1] * (e3 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N3[1] * N3[2] * vinf_arr2[2] * (e3 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - vinf_arr2[1] * np.cos(2 * true_anomaly) / np.sin(true_anomaly)))/(2 * (1 + e3**2 + 2 * e3 * np.cos(true_anomaly)))), -((np.sqrt(p3/mu3) * (vinf_arr2[2] - N3[1] * vinf_arr2[0] * (e3 + np.cos(true_anomaly)) / np.sin(true_anomaly) + N3[0] * vinf_arr2[1] * (e3 + np.cos(true_anomaly)) / np.sin(true_anomaly) + N3[0] * N3[2] * vinf_arr2[0] * (1/np.tan(true_anomaly) + e3 / np.sin(true_anomaly))**2 + N3[1] * N3[2] * vinf_arr2[1] * (1/np.tan(true_anomaly) + e3 / np.sin(true_anomaly))**2 + N3[2]**2 * vinf_arr2[2] * (1/np.tan(true_anomaly) + e3 / np.sin(true_anomaly))**2) * np.sin(true_anomaly))/(1 + e3**2 + 2 * e3 * np.cos(true_anomaly)))], dtype=np.longdouble)
    Q3 = np.cross(N3, P3)
    true_anomalySOI3 = np.arccos((p3/arrivalSOI - 1)/e3)
    diff3 = arrivalVector2
    arrivalVector2 = (p3 - arrivalSOI)/e3*P3 + arrivalSOI*np.sin(true_anomalySOI3)*Q3 + rplanets[arrivalIndex, arrival2] - rplanets[arrivalIndex, 0]
    diff3 = mag(arrivalVector2 - diff3)
    # print(diff0, diff1, diff2, diff3)
    # print("Difference in total : {}".format(diff0 + diff1 + diff2 + diff3))
    if diff0 + diff1 + diff2 + diff3 < 0.5:
        break
#Achtung: Verschiebung um 1 bei den indices!!!
tof0 = hyperbolicTOF(mu1, periapsisEarthOrbit/(1 - e1), e1, true_anomalySOI1)
tofarrival = abs(hyperbolicTOF(mu3, arrivalOrbitRadius/(1 - e3), e3, true_anomalySOI3))
DepartureTime = epoch + datetime.timedelta(seconds=float(t1 - tof0))
ArrivalTime = epoch + datetime.timedelta(seconds=float(t3 + tofarrival))
print("Thrust1:\nTime: {}\nHeliocentric position: {}\nTarget velocity: {}\n\u0394v: {} ({} km/s)".format(DepartureTime.strftime("%d.%m.%y.    %H:%M:%S"), periapsisEarthOrbit*P1 + rplanets[departureIndex, departure1] - rplanets[departureIndex, 0], np.sqrt(mu1/p1)*(e1 + 1)*Q1, (np.sqrt(mu1/p1)*(e1 + 1) - np.sqrt(mu1/periapsisEarthOrbit))*Q1, (np.sqrt(mu1/p1)*(e1 + 1) - np.sqrt(mu1/periapsisEarthOrbit))/1000))
print("Thrust2:\nTime: {}\nHeliocentric position: {}\nTarget velocity: {}\n\u0394v: {} ({} km/s)".format(ArrivalTime.strftime("%d.%m.%y.    %H:%M:%S"), arrivalOrbitRadius*P3 + rplanets[arrivalIndex, arrival2] - rplanets[arrivalIndex, 0], np.sqrt(mu3/p3)*(e3 + 1)*Q3, (np.sqrt(mu3/p3)*(e3 + 1) - np.sqrt(mu3/arrivalOrbitRadius))*Q3, (np.sqrt(mu3/p3)*(e3 + 1) - np.sqrt(mu3/arrivalOrbitRadius))/1000))
print("Total \u0394v required: {} km/s".format((np.sqrt(mu1/p1)*(e1 + 1) - np.sqrt(mu1/periapsisEarthOrbit) + np.sqrt(mu3/p3)*(e3 + 1) - np.sqrt(mu3/arrivalOrbitRadius))/1000))
print("The difference in v_infty magnitude at swingby: {}".format(abs(mag(vinf_dep2) - mag(vinf_arr1))))
print("Data for simulation :")
print("t0 =  {}".format(t1 - tof0))
print("t1 = {}".format(t3 + tofarrival))
print("rinit = np.array([\"{}\", \"{}\", \"{}\"], dtype=np.longdouble)".format(periapsisEarthOrbit*P1[0], periapsisEarthOrbit*P1[1], periapsisEarthOrbit*P1[2]))
rv = np.sqrt(mu1/p1)*(e1 + 1)*Q1
print("vinit = np.array([\"{}\", \"{}\", \"{}\"], dtype=np.longdouble)".format(rv[0], rv[1], rv[2]))
print("refpos = np.array([\"{}\", \"{}\", \"{}\"], dtype=np.longdouble)".format(arrivalOrbitRadius*P3[0], arrivalOrbitRadius*P3[1], arrivalOrbitRadius*P3[2]))
print("target = {}".format(x))