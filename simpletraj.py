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

def timeToIndex(t):
    return abs(int(round(t/dt)) + N2)

def hyperbolicTOF(mu, a, e, true_anomaly):#BUG CHECK
    print("a: {}".format((-a)**3))
    F = np.arccosh((e + np.cos(true_anomaly))/(1 + e*np.cos(true_anomaly)))
    print(e)
    print((e + np.cos(true_anomaly))/(1 + e*np.cos(true_anomaly)))
    print(F)
    print(e*np.sinh(F) - F)
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





mode0 = True



check = True
while check:
    print("What is your target ?\n1)  Mercury\n2)  Venus\n4)  Mars\n5)  Jupiter")
    x = input()
    try:
        x = int(x)
        if x < 1 or x > 5:
            print("Invalid index {} cannot be chosen".format(x))
        else:
            check = False
    except ValueError:
        print("Please provide an integer")

periapsisEarthOrbit = 185000 + 6378137      #According to Vallado: 
if mode0:
    departure = 3
    arrival = x
    mu0 = G*masses[departure]
    mu1 = G*masses[arrival]
    transfer_time = np.array([[np.nan, np.nan],[np.nan, np.nan],[100, 160],[0, 0], [120, 270], [2*365, 3*365]], dtype=np.longdouble)*86400
    transfer_range = transfer_time[x]
    search_range = transfer_range[1] - transfer_range[0]
    #Up to what time in the future do we want to check for possible launch dates?
    range_of_analysis = 20*31536000
    sd1 = int(range_of_analysis/tick_rate[departure])      #TEST FOR POSSIBLE BUG
    sd2 = int(int(search_range)/int(tick_rate[arrival])) + 1
    pchp1 = np.full([sd1, sd2, 1], np.nan, dtype=np.longdouble)
    jmax = 0
    i = 0
    for t1 in trange(int(deltat), int(deltat + range_of_analysis), int(tick_rate[departure])):
        j = 0
        for t2 in range(int(t1) + int(transfer_range[0]), int(t1) + int(transfer_range[1]), int(tick_rate[x])):
            tmp = porkchop(t1, t2, departure, arrival)
            if tmp is not None:
                C3, vinf_departure, vinf_arrival = tmp
                vinf_departure = mag(vinf_departure)
                vinf_arrival = mag(vinf_arrival)
                pchp1[i][j] = vinf_departure # + vinf_arrival
            else:
                tqdm.tqdm.write("oh")
                pchp1[i][j] = np.nan
            j += 1
        if i == 0:
            jmax = j - 1
        i += 1


    minindices = np.unravel_index(np.nanargmin(pchp1), pchp1.shape)


    departureTime = minindices[0]*tick_rate[departure] + deltat
    arrivalTime = minindices[1]*tick_rate[arrival] + transfer_range[0] + departureTime
    departureIndex = timeToIndex(departureTime)
    arrivalIndex = timeToIndex(arrivalTime)
    _, vinf_departure, vinf_arrival = porkchop(departureTime, arrivalTime, departure, arrival)

    # rdep = rplanets[departureIndex, departure] - rplanets[departureIndex, 0]
    # rarr = rplanets[arrivalIndex, arrival] - rplanets[arrivalIndex, 0]
    # vdep = vplanets[departureIndex, departure] - vplanets[departureIndex, 0] + vinf_departure
    # varr = vplanets[arrivalIndex, arrival] - vplanets[arrivalIndex, 0] + vinf_arrival
    
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

    departureSOI = mag(rplanets[departureIndex, departure] - rplanets[departureIndex, 0])*(masses[departure]/masses[0])**(2/5)
    arrivalSOI = mag(rplanets[arrivalIndex, arrival] - rplanets[arrivalIndex, 0])*(masses[arrival]/masses[0])**(2/5)
    print("Input arrival orbit radius")
    while True:
        arrivalOrbitRadius = input()
        try:
            arrivalOrbitRadius = np.longdouble(arrivalOrbitRadius)
            if arrivalOrbitRadius > arrivalSOI:
                print("Specified radius greater than the SOI of {}".format(names[arrival]))
            elif arrivalOrbitRadius < rad_of_closest_approach[arrival]:
                print("Specified radius is less than the lower limit. Do you want to use this radius nonetheless? (y/n)")
                if input().capitalize() == "Y":
                    break
            else:
                break
        except ValueError:
            print("Please input a number")


    rotationVectorEarth = earthUnitRotationVector(departureTime)
    rotationVectorTarget = unitRotationVector(arrivalTime, x)
    earthAtDeparture = rplanets[departureIndex, departure] - rplanets[departureIndex, 0]
    targetAtArrival = rplanets[arrivalIndex, arrival] - rplanets[arrivalIndex, 0]
    #TODO
    s = - np.sum(vinf_departure*rotationVectorEarth)/np.sum(vinf_departure**2)
    N = norm(rotationVectorEarth + s*vinf_departure)
    e = mag(vinf_departure)**2*periapsisEarthOrbit/(mu0) + 1        #See notes made in Vallado
    p = (mag(vinf_departure)**2*periapsisEarthOrbit**2/(mu0) + 2*periapsisEarthOrbit)        #See notes made in Vallado
    true_anomaly = np.arccos(-1/e)     #@infinity
    P = np.array([1/(1 + e**2 + 2 * e * np.cos(true_anomaly)) * np.sqrt(p/mu0) * (N[2] * vinf_departure[1] * (e + np.cos(true_anomaly)) - N[1] * vinf_departure[2] * (e + np.cos(true_anomaly)) - 2 * e * vinf_departure[0] / np.tan(true_anomaly) - vinf_departure[0] / np.sin(true_anomaly) - e**2 * vinf_departure[0] / np.sin(true_anomaly) + N[1]**2 * vinf_departure[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + N[2]**2 * vinf_departure[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N[0] * N[1] * vinf_departure[1] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N[0] * N[2] * vinf_departure[2] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly)), -((np.sqrt(p/mu0) * (2 * N[2] * vinf_departure[0] * (e + np.cos(true_anomaly)) - 2 * N[0] * vinf_departure[2] * (e + np.cos(true_anomaly)) + vinf_departure[1] / np.sin(true_anomaly) + 2 * N[0] * N[1] * vinf_departure[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N[1]**2 * vinf_departure[1] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N[1] * N[2] * vinf_departure[2] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - vinf_departure[1] * np.cos(2 * true_anomaly) / np.sin(true_anomaly)))/(2 * (1 + e**2 + 2 * e * np.cos(true_anomaly)))), -((np.sqrt(p/mu0) * (vinf_departure[2] - N[1] * vinf_departure[0] * (e + np.cos(true_anomaly)) / np.sin(true_anomaly) + N[0] * vinf_departure[1] * (e + np.cos(true_anomaly)) / np.sin(true_anomaly) + N[0] * N[2] * vinf_departure[0] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2 + N[1] * N[2] * vinf_departure[1] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2 + N[2]**2 * vinf_departure[2] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2) * np.sin(true_anomaly))/(1 + e**2 + 2 * e * np.cos(true_anomaly)))], dtype=np.longdouble)
    print("TEST 1: VALUE OUGHT TO BE NEAR 0: {}".format(N.dot(P)))
    print("TEST 2: Value ought to be near 1: {}".format(mag(P)))
    Q = np.cross(N, P)
    true_anomalySOI = np.arccos((p/departureSOI - 1)/e)
    departureVector = (p - departureSOI)/e*P + departureSOI*np.sin(true_anomalySOI)*Q + rplanets[departureIndex, departure] - rplanets[departureIndex, 0]

    s = - np.sum(vinf_arrival*rotationVectorTarget)/np.sum(vinf_arrival**2)
    N = (rotationVectorTarget + s*vinf_arrival)/mag(rotationVectorTarget + s*vinf_arrival)
    print("Test: Value ought to be near 0: {}".format(N.dot(vinf_arrival)))
    e = mag(vinf_arrival)**2*arrivalOrbitRadius/(mu1) + 1
    p = (mag(vinf_arrival)**2*arrivalOrbitRadius**2/(mu1) + 2*arrivalOrbitRadius)
    true_anomaly = -np.arccos(-1/e)
    P = np.array([1/(1 + e**2 + 2 * e * np.cos(true_anomaly)) * np.sqrt(p/mu1) * (N[2] * vinf_arrival[1] * (e + np.cos(true_anomaly)) - N[1] * vinf_arrival[2] * (e + np.cos(true_anomaly)) - 2 * e * vinf_arrival[0] / np.tan(true_anomaly) - vinf_arrival[0] / np.sin(true_anomaly) - e**2 * vinf_arrival[0] / np.sin(true_anomaly) + N[1]**2 * vinf_arrival[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + N[2]**2 * vinf_arrival[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N[0] * N[1] * vinf_arrival[1] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N[0] * N[2] * vinf_arrival[2] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly)), -((np.sqrt(p/mu1) * (2 * N[2] * vinf_arrival[0] * (e + np.cos(true_anomaly)) - 2 * N[0] * vinf_arrival[2] * (e + np.cos(true_anomaly)) + vinf_arrival[1] / np.sin(true_anomaly) + 2 * N[0] * N[1] * vinf_arrival[0] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N[1]**2 * vinf_arrival[1] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N[1] * N[2] * vinf_arrival[2] * (e + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - vinf_arrival[1] * np.cos(2 * true_anomaly) / np.sin(true_anomaly)))/(2 * (1 + e**2 + 2 * e * np.cos(true_anomaly)))), -((np.sqrt(p/mu1) * (vinf_arrival[2] - N[1] * vinf_arrival[0] * (e + np.cos(true_anomaly)) / np.sin(true_anomaly) + N[0] * vinf_arrival[1] * (e + np.cos(true_anomaly)) / np.sin(true_anomaly) + N[0] * N[2] * vinf_arrival[0] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2 + N[1] * N[2] * vinf_arrival[1] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2 + N[2]**2 * vinf_arrival[2] * (1/np.tan(true_anomaly) + e / np.sin(true_anomaly))**2) * np.sin(true_anomaly))/(1 + e**2 + 2 * e * np.cos(true_anomaly)))], dtype=np.longdouble)
    Q = np.cross(N, P)
    true_anomalySOI = -np.arccos((p/arrivalSOI - 1)/e)
    arrivalVector = (p - arrivalSOI)/e*P + arrivalSOI*np.sin(true_anomalySOI)*Q + rplanets[arrivalIndex, arrival] - rplanets[arrivalIndex, 0]
    while True:
        vinf_departure, vinf_arrival = porkchop(departureTime, arrivalTime, departure, arrival, departureVector, arrivalVector)
        s = - np.sum(vinf_departure*rotationVectorEarth)/np.sum(vinf_departure**2)
        N0 = (rotationVectorEarth + s*vinf_departure)/mag(rotationVectorEarth + s*vinf_departure)
        e0 = mag(vinf_departure)**2*periapsisEarthOrbit/(mu0) + 1
        p0 = (mag(vinf_departure)**2*periapsisEarthOrbit**2/(mu0) + 2*periapsisEarthOrbit)
        true_anomaly = np.arccos(-1/e0)
        P0 = np.array([1/(1 + e0**2 + 2 * e0 * np.cos(true_anomaly)) * np.sqrt(p0/mu0) * (N0[2] * vinf_departure[1] * (e0 + np.cos(true_anomaly)) - N0[1] * vinf_departure[2] * (e0 + np.cos(true_anomaly)) - 2 * e0 * vinf_departure[0] / np.tan(true_anomaly) - vinf_departure[0] / np.sin(true_anomaly) - e0**2 * vinf_departure[0] / np.sin(true_anomaly) + N0[1]**2 * vinf_departure[0] * (e0 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + N0[2]**2 * vinf_departure[0] * (e0 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N0[0] * N0[1] * vinf_departure[1] * (e0 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N0[0] * N0[2] * vinf_departure[2] * (e0 + np.cos(true_anomaly))**2 / np.sin(true_anomaly)), -((np.sqrt(p0/mu0) * (2 * N0[2] * vinf_departure[0] * (e0 + np.cos(true_anomaly)) - 2 * N0[0] * vinf_departure[2] * (e0 + np.cos(true_anomaly)) + vinf_departure[1] / np.sin(true_anomaly) + 2 * N0[0] * N0[1] * vinf_departure[0] * (e0 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N0[1]**2 * vinf_departure[1] * (e0 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N0[1] * N0[2] * vinf_departure[2] * (e0 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - vinf_departure[1] * np.cos(2 * true_anomaly) / np.sin(true_anomaly)))/(2 * (1 + e0**2 + 2 * e0 * np.cos(true_anomaly)))), -((np.sqrt(p0/mu0) * (vinf_departure[2] - N0[1] * vinf_departure[0] * (e0 + np.cos(true_anomaly)) / np.sin(true_anomaly) + N0[0] * vinf_departure[1] * (e0 + np.cos(true_anomaly)) / np.sin(true_anomaly) + N0[0] * N0[2] * vinf_departure[0] * (1/np.tan(true_anomaly) + e0 / np.sin(true_anomaly))**2 + N0[1] * N0[2] * vinf_departure[1] * (1/np.tan(true_anomaly) + e0 / np.sin(true_anomaly))**2 + N0[2]**2 * vinf_departure[2] * (1/np.tan(true_anomaly) + e0 / np.sin(true_anomaly))**2) * np.sin(true_anomaly))/(1 + e0**2 + 2 * e0 * np.cos(true_anomaly)))], dtype=np.longdouble)
        print("TEST 1: VALUE OUGHT TO BE NEAR 0: {}".format(N0.dot(P0)))
        Q0 = np.cross(N0, P0)
        true_anomalySOI0 = np.arccos((p0/departureSOI - 1)/e0)
        diff1 = departureVector
        geocentricV = (p0 - departureSOI)/e0*P0 + departureSOI*np.sin(true_anomalySOI0)*Q0
        departureVector = geocentricV + rplanets[departureIndex, departure] - rplanets[departureIndex, 0]
        diff1 = mag(diff1 - departureVector)

        s = - np.sum(vinf_arrival*rotationVectorTarget)/np.sum(vinf_arrival**2)
        N1 = (rotationVectorTarget + s*vinf_arrival)/mag(rotationVectorTarget + s*vinf_arrival)
        e1 = mag(vinf_arrival)**2*arrivalOrbitRadius/(mu1) + 1
        p1 = (mag(vinf_arrival)**2*arrivalOrbitRadius**2/(mu1) + 2*arrivalOrbitRadius)
        true_anomaly = -np.arccos(-1/e1)
        P1 = np.array([1/(1 + e1**2 + 2 * e1 * np.cos(true_anomaly)) * np.sqrt(p1/mu1) * (N1[2] * vinf_arrival[1] * (e1 + np.cos(true_anomaly)) - N1[1] * vinf_arrival[2] * (e1 + np.cos(true_anomaly)) - 2 * e1 * vinf_arrival[0] / np.tan(true_anomaly) - vinf_arrival[0] / np.sin(true_anomaly) - e1**2 * vinf_arrival[0] / np.sin(true_anomaly) + N1[1]**2 * vinf_arrival[0] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + N1[2]**2 * vinf_arrival[0] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N1[0] * N1[1] * vinf_arrival[1] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - N1[0] * N1[2] * vinf_arrival[2] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly)), -((np.sqrt(p1/mu1) * (2 * N1[2] * vinf_arrival[0] * (e1 + np.cos(true_anomaly)) - 2 * N1[0] * vinf_arrival[2] * (e1 + np.cos(true_anomaly)) + vinf_arrival[1] / np.sin(true_anomaly) + 2 * N1[0] * N1[1] * vinf_arrival[0] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N1[1]**2 * vinf_arrival[1] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) + 2 * N1[1] * N1[2] * vinf_arrival[2] * (e1 + np.cos(true_anomaly))**2 / np.sin(true_anomaly) - vinf_arrival[1] * np.cos(2 * true_anomaly) / np.sin(true_anomaly)))/(2 * (1 + e1**2 + 2 * e1 * np.cos(true_anomaly)))), -((np.sqrt(p1/mu1) * (vinf_arrival[2] - N1[1] * vinf_arrival[0] * (e1 + np.cos(true_anomaly)) / np.sin(true_anomaly) + N1[0] * vinf_arrival[1] * (e1 + np.cos(true_anomaly)) / np.sin(true_anomaly) + N1[0] * N1[2] * vinf_arrival[0] * (1/np.tan(true_anomaly) + e1 / np.sin(true_anomaly))**2 + N1[1] * N1[2] * vinf_arrival[1] * (1/np.tan(true_anomaly) + e1 / np.sin(true_anomaly))**2 + N1[2]**2 * vinf_arrival[2] * (1/np.tan(true_anomaly) + e1 / np.sin(true_anomaly))**2) * np.sin(true_anomaly))/(1 + e1**2 + 2 * e1 * np.cos(true_anomaly)))], dtype=np.longdouble)
        Q1 = np.cross(N1, P1)
        true_anomalySOI1 = -np.arccos((p1/arrivalSOI - 1)/e1)
        diff2 = arrivalVector
        planetocentricV = (p1 - arrivalSOI)/e1*P1 + arrivalSOI*np.sin(true_anomalySOI1)*Q1
        arrivalVector = planetocentricV + rplanets[arrivalIndex, arrival] - rplanets[arrivalIndex, 0]
        if np.isnan(departureVector).any() or np.isnan(arrivalVector).any():
            print("NaN encounterd. Terminating program")
            exit(1)
        diff2 = mag(arrivalVector - diff2)
        if diff1 + diff2 < 0.5:
            break
    tof0 = hyperbolicTOF(mu0, periapsisEarthOrbit/(1 - e0), e0, true_anomalySOI0)
    tof1 = abs(hyperbolicTOF(mu1, arrivalOrbitRadius/(1 - e1), e1, true_anomalySOI1))
    DepartureTime = epoch + datetime.timedelta(seconds=float(departureTime - tof0))
    ArrivalTime = epoch + datetime.timedelta(seconds=float(arrivalTime + tof1))
    vdep = np.sqrt(mu0/p0)*(e0 + np.cos(0))*Q0
    print("Thrust1:\nTime: {}\nPlanetocentric position: {}\nTarget velocity: {}\n\u0394v: {} ({} km/s)".format(DepartureTime.strftime("%d.%m.%y.    %H:%M:%S"), periapsisEarthOrbit*P0, np.sqrt(mu0/p0)*(e0 + np.cos(0))*Q0, (np.sqrt(mu0/p0)*(e0 + 1) - np.sqrt(mu0/periapsisEarthOrbit))*Q0, (np.sqrt(mu0/p0)*(e0 + 1) - np.sqrt(mu0/periapsisEarthOrbit))/1000))
    print("Thrust2:\nTime: {}\nPlanetocentric position: {}\nTarget velocity: {}\n\u0394v: {} ({} km/s)".format(ArrivalTime.strftime("%d.%m.%y.    %H:%M:%S"), arrivalOrbitRadius*P1, np.sqrt(mu1/p1)*(e1 + np.cos(0))*Q1, (np.sqrt(mu1/p1)*(e1 + 1) - np.sqrt(mu1/arrivalOrbitRadius))*Q1, (np.sqrt(mu1/p1)*(e1 + 1) - np.sqrt(mu1/arrivalOrbitRadius))/1000))
    print("Total \u0394v required: {} km/s".format((np.sqrt(mu0/p0)*(e0 + 1) - np.sqrt(mu0/periapsisEarthOrbit) + np.sqrt(mu1/p1)*(e1 + 1) - np.sqrt(mu1/arrivalOrbitRadius))/1000))
    print("Data for simulation :")
    print("t0 = {}".format(departureTime - tof0))
    print("t1 = {}".format(arrivalTime + tof1))
    print("rinit = np.array([\"{}\", \"{}\", \"{}\"], dtype=np.longdouble)".format(periapsisEarthOrbit*P0[0], periapsisEarthOrbit*P0[1], periapsisEarthOrbit*P0[2]))
    rv = np.sqrt(mu0/p0)*(e0 + 1)*Q0
    print("vinit = np.array([\"{}\", \"{}\", \"{}\"], dtype=np.longdouble)".format(rv[0], rv[1], rv[2]))
    print("refpos = np.array([\"{}\", \"{}\", \"{}\"], dtype=np.longdouble)".format(arrivalOrbitRadius*P1[0], arrivalOrbitRadius*P1[1], arrivalOrbitRadius*P1[2]))
    print("target = {}".format(x))