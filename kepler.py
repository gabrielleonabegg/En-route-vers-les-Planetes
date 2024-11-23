from math import sqrt, factorial, pi
import numpy as np

def vect(a,b,c):
    return np.array([a,b,c], dtype=np.longdouble)

def mag(a:np.ndarray):
    return np.sqrt(a.dot(a))            #https://stackoverflow.com/questions/9171158/how-do-you-get-the-magnitude-of-a-vector-in-numpy

def C(z):
    s = 0
    a = 0.5
    k = 1
    while abs(a) > 0.00000000001:
        s += a
        a = (-z)**k / np.longdouble(factorial(2*k+2))
        k +=1
    return s

def S(z):
    s = 0
    a = 1/6
    k = 1
    while abs(a) > 0.00000000001:
        s += a
        a = (-z)**k / np.longdouble(factorial(2*k+3))
        k +=1
    return s

def kepler(r0v: np.ndarray, v0v: np.ndarray, t, mu, xn = None):
    if t == 0:      #enke's method might make use of this
        return r0v, v0v
    r0 = mag(r0v)
    v0 = mag(v0v)
    E = 0.5*(v0)**2 - mu/r0
    a = - mu/(2*E)
    h = np.cross(r0v, v0v)
    alpha = (2*mu/r0 - v0**2)/mu    #a^-1 to make sure we don't divide by 0, as suggested on page 169, eq (4-74)
    e = sqrt(1+(2*E*mag(h)**2)/(mu)**2)
    if 0 < e < 1 and mu != 0:
        TP = 2*np.pi*np.sqrt(a**3/mu)
        while t > TP:
            t = t - TP
        while t < 0:
            t += TP
    #solve for x when time is known: section 4.4.2
    if xn == None:
        #select starting value
        if 0 < e < 1:
            xn = sqrt(mu)*t*alpha
        else:
            xn = np.sign(t)*np.sqrt(-a)*np.log((-2*mu*t)/(a*(r0v.dot(v0v) + np.sign(t)*np.sqrt(-mu*a)*(1-r0/a))))
        tn = t+20
    #determining x from initial conditions and time (Newton iteration scheme)
    while abs((t-tn)/t) > 1e-8: #in our elliptic example, it doesn't even get executed once!
        #print("Goal: {}".format(t))
        z = xn**2 *alpha
        tn = np.dot(r0v, v0v)/mu * xn**2 * C(z) + (1-r0*alpha)* xn**3*S(z) / sqrt(mu) + r0*xn/sqrt(mu)
        #print(tn)
        dtdx = (xn**2*C(z) + np.dot(r0v, v0v)/sqrt(mu) * xn*(1 - z*S(z)) + r0*(1 - z*C(z)))/sqrt(mu)
        xn += (t-tn)/dtdx
    x = xn
    z = x**2 *alpha
    # the f and g expression
    f = 1 - x**2/r0 * C(z)
    g = t - x**3/sqrt(mu) * S(z)
    # getting position vector rv and its absolute value r (for further computation of v)
    rv = r0v*f + v0v*g
    r = mag(rv)
    dg = 1 - x**2/r * C(z)
    df = sqrt(mu)/r0/r *x*(z*S(z) - 1)
    #getting vector v
    v = r0v*df + v0v*dg
    #precision should be 1
    precision = f*dg + df * g
    print("Precision in the Kepler equation: {}".format(precision))
    return rv, v

"""
G = 6.674e-11
class object:
    def __init__(self, r: vect, v: vect, m):
        self.r = r
        self.v = v
        self.m = m

class planet(object):
    def __init__(self, r: vect, v: vect, m, rayon):
        super().__init__(r, v, m)
        self.rayon = rayon


earth = planet(vect(0,0,0), vect(0,0,0), 5.9722e24, 6.371e6)    #Idee hinter den Startbedingungen: Siehe Arbeitsjournal
satellite = object(vect(6478100,0, 0), vect(0, 10000, 0), 2500)
mu = G*earth.m

E = 0.5*mag(satellite.v)**2 - mu/mag(satellite.r - earth.r)
h = np.cross(satellite.r, satellite.v)
a = - mu/(2*E)
TP = 2*pi*sqrt(a**3/mu)
p = mag(h)**2/(G*earth.m)
e = sqrt(1+(2*E*mag(h)**2)/(G*earth.m)**2)
ra = p/(1 - e)

rk, vk = kepler(satellite.r, satellite.v, -TP/2, mu)
print(abs(ra - mag(rk)))
"""