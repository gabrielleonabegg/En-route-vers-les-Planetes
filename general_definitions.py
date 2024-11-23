import numpy as np
import datetime
import os
from helpers1 import mag

a = np.array([[3250433/53222400, 572741/5702400, -8701681/39916800, 4026311/13305600, -917039/3193344, 7370669/39916800, -1025779/13305600, 754331/39916800, -330157/159667200],
 [-330157/159667200, 530113/6652800, 518887/19958400, -27631/623700, 44773/1064448, -531521/19958400, 109343/9979200, -1261/475200, 45911/159667200],
 [45911/159667200, -185839/39916800, 171137/1900800, 73643/39916800, -25775/3193344, 77597/13305600, -98911/39916800, 24173/39916800, -3499/53222400],
 [-3499/53222400, 4387/4989600, -35039/4989600, 90817/950400, -20561/3193344, 2117/9979200, 2059/6652800, -317/2851200, 317/22809600],
 [317/22809600, -2539/13305600, 55067/39916800, -326911/39916800, 14797/152064, -326911/39916800, 55067/39916800, -2539/13305600, 317/22809600],
 [317/22809600, -317/2851200, 2059/6652800, 2117/9979200, -20561/3193344, 90817/950400, -35039/4989600, 4387/4989600, -3499/53222400],
 [-3499/53222400, 24173/39916800, -98911/39916800, 77597/13305600, -25775/3193344, 73643/39916800, 171137/1900800, -185839/39916800, 45911/159667200],
 [45911/159667200, -1261/475200, 109343/9979200, -531521/19958400, 44773/1064448, -27631/623700, 518887/19958400, 530113/6652800, -330157/159667200],
 [-330157/159667200, 754331/39916800, -1025779/13305600, 7370669/39916800, -917039/3193344, 4026311/13305600, -8701681/39916800, 572741/5702400, 3250433/53222400],
 [3250433/53222400, -11011481/19958400, 6322573/2851200, -8660609/1663200, 25162927/3193344, -159314453/19958400, 18071351/3326400, -24115843/9979200, 103798439/159667200]], dtype=np.longdouble)

b = np.asarray([[19087/89600, -427487/725760, 3498217/3628800, -500327/403200, 6467/5670, -2616161/3628800, 24019/80640, -263077/3628800, 8183/1036800],
 [8183/1036800, 57251/403200, -1106377/3628800, 218483/725760, -69/280, 530177/3628800, -210359/3628800, 5533/403200, -425/290304],
 [-425/290304, 76453/3628800, 5143/57600, -660127/3628800, 661/5670, -4997/80640, 83927/3628800, -19109/3628800, 7/12800],
 [7/12800, -23173/3628800, 29579/725760, 2497/57600, -2563/22680, 172993/3628800, -6463/403200, 2497/725760, -2497/7257600],
 [-2497/7257600, 1469/403200, -68119/3628800, 252769/3628800, 0, -252769/3628800, 68119/3628800, -1469/403200, 2497/7257600],
 [2497/7257600, -2497/725760, 6463/403200, -172993/3628800, 2563/22680, -2497/57600, -29579/725760, 23173/3628800, -7/12800],
 [-7/12800, 19109/3628800, -83927/3628800, 4997/80640, -661/5670, 660127/3628800, -5143/57600, -76453/3628800, 425/290304],
 [425/290304, -5533/403200, 210359/3628800, -530177/3628800, 69/280, -218483/725760, 1106377/3628800, -57251/403200, -8183/1036800],
 [-8183/1036800, 263077/3628800, -24019/80640, 2616161/3628800, -6467/5670, 500327/403200, -3498217/3628800, 427487/725760, -19087/89600],
 [25713/89600, -9401029/3628800, 5393233/518400, -9839609/403200, 167287/4536, -135352319/3628800, 10219841/403200, -40987771/3628800, 3288521/1036800]], dtype=np.longdouble)

N = 8
N2 = int(N/2)

T = 5200848000
dt = 1000       #2 Stunden

# 1 Jahr: 31558140
# 165 Jahre: 5200848000

steps = int(T/dt) + N2 + 2  #+1 for the initial value + 1 because int always rounds off. If it doesent, because T is a multiple of dt, then "while t <= T" will be true for the last iteration, as we needn't worry for floatingpoint imprecision (dt is an int)
masses = np.array([1.988410e30, 3.302e23, 4.8685e24, 6.04568e24, 6.4171e23, 1.89858032e27, 5.684805395e26, 8.6821876e25, 1.0243e26], dtype=np.longdouble)       # Für Jupiter baryzentrum : https://nssdc.gsfc.nasa.gov/planetary/factsheet/galileanfact_table.html; https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
objcount = len(masses)

#Merkur : Vernachlässigbare Atmosphäre; Der minimale Radius wurde ungefähr beim Radius des Äquators angesetzt.
#Venus : https://nssdc.gsfc.nasa.gov/planetary/factsheet/venusfact.html: Scale height: 15.9 km, Surface density: ~65. kg/m3 -> gemäss Idealem Gasgesetz kann bei konstanter Temperatur (eine unserer vereinfachenden Annahmen) die Gleichung h = -H*(ln(p) - ln(p0)) angewendet werden. Wir gehen davon aus, dass ab einer Dichte von 1e-8 kg/m^3 die Atmosphäre vernachlässigbar ist. Dies würde auf der Erde etwa einer Distanz von 120-130km entsprechen. Auf der Venus h = -15900*(ln(1e-8) - ln(65.)) = 359262m . Äquatorial radius: 6051.893km = 6051893 m. Also: 6.41116e6 m
#Erde : 6.49814e6 (6378.137km radius am Äquator + 120 km Atmosphäre)
#Mars : Same calculations as for Venus with Surface density: ~0.020 kg/m3 and Scale height:  11.1 km (Source : https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) rpmin = 3.55724e6
#Jupiter : " During the  Jupiter 
# gravity  assists,  the minimum  allowed  perisapse  radius  was  set  to  1.1  Jovian  radii.   For  comparison,  Pioneer  11 
# approached  to  1.6  Jovian  radii  during  its  flyby  in  December  1974,  and  the  Juno  spacecraft  is  expected  to  orbit 
# Jupiter with a periapse of approximately 1.07 Jovian radii.   This is a critical issue because  of the extreme  magnetic 
# field  surrounding  Jupiter  and  the intense  radiation  environment.  A  close  approach  will  run  the  risk of  damage  to 
# electronics or will require the addition of shielding materials. " https://www.researchgate.net/publication/268556821_A_Survey_of_Mission_Opportunities_to_Trans-Neptunian_Objects_-_Part_II_Orbital_Capture?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoiX2RpcmVjdCJ9fQ (p. 4)    1 Jovian Radius according to https://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html is 71,398 km = 71,398,000 m which yields a minimal periapsis height = 7.85378e7
#Saturn: End of Saturn's A-Ring: 136775000 m = 1.36775000e8 m
#Uranus
#Neptun

rad_of_closest_approach = np.array([np.nan, 2.4405e6,6.400248e6, 6.500137e6, 3.5497e6, 7.86412e7, 4.8e8, 2.6026390e7, 2.5107930e7], dtype=np.longdouble)#Voir le tableau \ref
tick_rate = [None, 5*60*60, 1*86400, 2*86400, 5*86400, 6*86400, 7*86400, 10*86400, 14*86400]     #Arbitrary, earth and mars inspired by Wikipedia

names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

def pEpoch():
    with open("log.txt", "r") as file:
        ep = float(file.read())
    return datetime.datetime.fromtimestamp(ep)


def unitRotationVector(t, planet:int):      #Report of the IAU Working Group on Cartographic Coordinates and Rotational Elements: 2015
    if planet == 3:
        return earthUnitRotationVector(t)
    t = float(t)
    T = (datetime.timedelta(seconds=t) + pEpoch() - datetime.datetime(year=2000, month=1, day=1, hour=12, minute=0, second=0, microsecond=0)).total_seconds()/(36525*24*60*60)
    if planet == 1:
        alpha0 = 281.0103 - 0.0328*T
        gamma0 = 61.4155 - 0.0049*T
    elif planet == 2:
        alpha0 = 272.76
        gamma0 = 67.16
    elif planet == 4:
        alpha0 = 317.269202 - 0.10927547*T+ 0.000068 * np.sin((198.991226 + 19139.4819985*T )*np.pi/180) + 0.000238 * np.sin((226.292679 + 38280.8511281*T)*np.pi/180) + 0.000052 * np.sin((249.663391 + 57420.7251593*T)*np.pi/180) + 0.000009 *np.sin((266.183510 + 76560.6367950*T)*np.pi/180) + 0.419057 * np.sin((79.398797 + 0.5042615*T )*np.pi/180)
        gamma0 = 54.432516 - 0.05827105*T + 0.000051 * np.cos((122.433576 + 19139.9407476*T )*np.pi/180) + 0.000141 * np.cos((43.058401 + 38280.8753272*T )*np.pi/180) + 0.000031 * np.cos((57.663379 + 57420.7517205*T )*np.pi/180) + 0.000005 *np.cos((79.476401 + 76560.6495004*T )*np.pi/180) + 1.591274 * np.cos((166.325722 + 0.5042615*T)*np.pi/180)
    elif planet == 5:
        J = np.array([99.360714 + 4850.4046*T, 175.895369 + 1191.9605*T, 300.323162 + 262.5475*T, 114.012305 + 6070.2476*T, 49.511251 + 64.3000*T])*np.pi/180
        alpha0 = 268.056595 - 0.006499*T + 0.000117*np.sin(J[0]) + 0.000938 * np.sin(J[1]) + 0.001432* np.sin(J[2]) + 0.000030* np.sin(J[3]) + 0.002150 *np.sin(J[4])
        gamma0 = 64.495303 + 0.002413*T + 0.000050* np.cos(J[0]) + 0.000404 * np.cos (J[1]) + 0.000617 * np.cos (J[2]) - 0.000013 * np.cos(J[3]) + 0.000926 * np.cos(J[4])
    elif planet == 6:
        alpha0 = 40.589 - 0.036*T
        gamma0 = 83.537 - 0.004*T
    elif planet == 7:
        alpha0 = 257.311
        gamma0 = -15.175
    elif planet == 8:
        N = (357.85 + 52.316*T)*np.pi/180
        alpha0 = 299.36 + 0.70 * np.sin(N)
        gamma0 = 43.46 - 0.51 * np.cos(N)

    alpha0 = alpha0*np.pi/180
    gamma0 = gamma0*np.pi/180
    if planet != 2 and planet != 7:#Voir la distinction entre le mouvement prograde ou rétrograde fait dans \cite IAU p. 
        w = np.array([np.cos(alpha0)*np.cos(gamma0), np.sin(alpha0)*np.cos(gamma0), np.sin(gamma0)], dtype=np.longdouble)#\cite p.45 FoA
    else:
        w = - np.array([np.cos(alpha0)*np.cos(gamma0), np.sin(alpha0)*np.cos(gamma0), np.sin(gamma0)], dtype=np.longdouble)
    return w/mag(w)

def earthUnitRotationVector(t1):        #Report of the IAU Working Group on Cartographic Coordinates and Rotational Elements: 2009
    t1 = float(t1)
    T = (datetime.timedelta(seconds=t1) + pEpoch() - datetime.datetime(year=2000, month=1, day=1, hour=12, minute=0, second=0, microsecond=0)).total_seconds()/(36525*24*60*60)
    alpha = np.longdouble(0.00 - 0.641*T)*(np.pi/180)
    gamma = np.longdouble(90 - 0.557*T)*np.pi/180
    w = np.array([np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.cos(gamma), np.sin(gamma)], dtype=np.longdouble)#\cite p.45 FoA
    return w/mag(w)

