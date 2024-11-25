import numpy as np
from helpers1 import *
from general_definitions import *
import datetime
from tqdm import tqdm
from math import ceil

Nold = N
N2old = N2

b = [[94815908183/402361344000, -307515172843/373621248000, 2709005666077/1307674368000, -2309296746931/523069747200, 507942835493/69742632960, -4007043002299/435891456000, 2215533/250250, -2816016533573/435891456000, 175102023617/49816166400, -144690945961/104613949440, 486772076771/1307674368000, -160495253651/2615348736000, 2224234463/475517952000],
[2224234463/475517952000, 7034932909/40236134400, -599204812637/1307674368000, 18278957351/24908083200, -373290894521/348713164800, 550602114107/435891456000, -4538591/3891888, 360347741893/435891456000, -153580740679/348713164800, 89210842829/523069747200, -8468059909/186810624000, 3869513783/523069747200, -4012317/7175168000],
[-4012317/7175168000, 31245653651/2615348736000, 1199987603/9144576000, -31205504599/104613949440, 116481399481/348713164800, -21844230061/62270208000, 18461231/60810750, -90050181701/435891456000, 7462997467/69742632960, -7078368623/174356582400, 13891101923/1307674368000, -4478654099/2615348736000, 13681829/106748928000],
[13681829/106748928000, -5820152083/2615348736000, 5739162887/261534873600, 345913117/3657830400, -72062156729/348713164800, 73700317499/435891456000, -261983/2002000, 1041757943/12454041600, -14518999879/348713164800, 8038193101/523069747200, -5153476771/1307674368000, 148748057/237758976000, -1935865/41845579776],
[-1935865/41845579776, 173463193/237758976000, -1089820997/186810624000, 6133014383/174356582400, 149947703/2438553600, -12825001151/87178291200, 5454343/60810750, -22437447749/435891456000, 8407070279/348713164800, -639529483/74724249600, 558737863/261534873600, -869611667/2615348736000, 521303/21525504000],
[521303/21525504000, -944389651/2615348736000, 3424241827/1307674368000, -6674450381/523069747200, 522979469/9963233280, 92427157/3048192000, -51350723/486486000, 20981972677/435891456000, -7081103431/348713164800, 236881763/34871316480, -2134386979/1307674368000, 92427157/373621248000, -92427157/5230697472000],
[-92427157/5230697472000, 132822967/523069747200, -2274524387/1307674368000, 4013113421/523069747200, -8855328071/348713164800, 32793164357/435891456000, 0, -32793164357/435891456000, 8855328071/348713164800, -4013113421/523069747200, 2274524387/1307674368000, -132822967/523069747200, 92427157/5230697472000],
[92427157/5230697472000, -92427157/373621248000, 2134386979/1307674368000, -236881763/34871316480, 7081103431/348713164800, -20981972677/435891456000, 51350723/486486000, -92427157/3048192000, -522979469/9963233280, 6674450381/523069747200, -3424241827/1307674368000, 944389651/2615348736000, -521303/21525504000],
[-521303/21525504000, 869611667/2615348736000, -558737863/261534873600, 639529483/74724249600, -8407070279/348713164800, 22437447749/435891456000, -5454343/60810750, 12825001151/87178291200, -149947703/2438553600, -6133014383/174356582400, 1089820997/186810624000, -173463193/237758976000, 1935865/41845579776],
[1935865/41845579776, -148748057/237758976000, 5153476771/1307674368000, -8038193101/523069747200, 14518999879/348713164800, -1041757943/12454041600, 261983/2002000, -73700317499/435891456000, 72062156729/348713164800, -345913117/3657830400, -5739162887/261534873600, 5820152083/2615348736000, -13681829/106748928000],
[-13681829/106748928000, 4478654099/2615348736000, -13891101923/1307674368000, 7078368623/174356582400, -7462997467/69742632960, 90050181701/435891456000, -18461231/60810750, 21844230061/62270208000, -116481399481/348713164800, 31205504599/104613949440, -1199987603/9144576000, -31245653651/2615348736000, 4012317/7175168000],
[4012317/7175168000, -3869513783/523069747200, 8468059909/186810624000, -89210842829/523069747200, 153580740679/348713164800, -360347741893/435891456000, 4538591/3891888, -550602114107/435891456000, 373290894521/348713164800, -18278957351/24908083200, 599204812637/1307674368000, -7034932909/40236134400, -2224234463/475517952000],
[-2224234463/475517952000, 160495253651/2615348736000, -486772076771/1307674368000, 144690945961/104613949440, -175102023617/49816166400, 2816016533573/435891456000, -2215533/250250, 4007043002299/435891456000, -507942835493/69742632960, 2309296746931/523069747200, -2709005666077/1307674368000, 307515172843/373621248000, -94815908183/402361344000],
[106364763817/402361344000, -9000055832083/2615348736000, 491703913717/23775897600, -13247042672623/174356582400, 66393001798471/348713164800, -149831214658501/435891456000, 31975145483/69498000, -40318232897599/87178291200, 121844891963321/348713164800, -102675619234099/523069747200, 104639289835229/1307674368000, -59344946587373/2615348736000, 733526173/172204032]]

a = [[132282840127/2414168064000, 192413017/1162377216, -183706612697/348713164800, 133373184587/112086374400, -467089093853/232475443200, 26637354127/10378368000, -186038426051/74724249600, 5304463979/2905943040, -77220056327/77491814400, 308415783287/784604620800, -8800586233/83026944000, 1525695617/87178291200, -1197622087/896690995200],
[-1197622087/896690995200, 14516634431/201180672000, 64188105383/1046139494400, -227269902593/1569209241600, 23409520499/99632332800, -25305946559/87178291200, 102644956367/373621248000, -492615587/2490808320, 74251645873/697426329600, -65181504263/1569209241600, 11614474687/1046139494400, -679920221/373621248000, 866474507/6276836966400],
[866474507/6276836966400, -2046617/653837184, 3033240127/36578304000, 613021979/28021593600, -4596037109/99632332800, 2439013/42567525, -3989974979/74724249600, 2062196293/54486432000, -14026509859/697426329600, 380746409/49037788800, -12299341/5977939968, 446819/1334361600, -72041419/2853107712000],
[-72041419/2853107712000, 58072601/124540416000, -423410459/83026944000, 989217599/10973491200, 296244089/77491814400, -1980839447/145297152000, 5218829519/373621248000, -1462665121/145297152000, 177708673/33210777600, -65905289/32024678400, 946410977/1743565824000, -315501/3587584000, 207259963/31384184832000],
[207259963/31384184832000, -20754971/186810624000, 5133428761/5230697472000, -5483137573/784604620800, 462681151/4877107200, -145599917/31135104000, -859562191/373621248000, 574469327/217945728000, -156165377/99632332800, 493595561/784604620800, -885138967/5230697472000, 3292123/118879488000, -9374747/4483454976000],
[-9374747/4483454976000, 1606609/47551795200, -40978319/149448499200, 2478440803/1569209241600, -5916580259/697426329600, 297378623/3048192000, -123511513/14944849920, 2290603/1779148800, -38522503/697426329600, -16224893/224172748800, 162596491/5230697472000, -3203699/523069747200, 3203699/6276836966400],
[3203699/6276836966400, -1901831/217945728000, 5132899/69742632960, -164834207/392302310400, 452015111/232475443200, -22134407/2421619200, 36740617/373248000, -22134407/2421619200, 452015111/232475443200, -164834207/392302310400, 5132899/69742632960, -1901831/217945728000, 3203699/6276836966400],
[3203699/6276836966400, -3203699/523069747200, 162596491/5230697472000, -16224893/224172748800, -38522503/697426329600, 2290603/1779148800, -123511513/14944849920, 297378623/3048192000, -5916580259/697426329600, 2478440803/1569209241600, -40978319/149448499200, 1606609/47551795200, -9374747/4483454976000],
[-9374747/4483454976000, 3292123/118879488000, -885138967/5230697472000, 493595561/784604620800, -156165377/99632332800, 574469327/217945728000, -859562191/373621248000, -145599917/31135104000, 462681151/4877107200, -5483137573/784604620800, 5133428761/5230697472000, -20754971/186810624000, 207259963/31384184832000],
[207259963/31384184832000, -315501/3587584000, 946410977/1743565824000, -65905289/32024678400, 177708673/33210777600, -1462665121/145297152000, 5218829519/373621248000, -1980839447/145297152000, 296244089/77491814400, 989217599/10973491200, -423410459/83026944000, 58072601/124540416000, -72041419/2853107712000],
[-72041419/2853107712000, 446819/1334361600, -12299341/5977939968, 380746409/49037788800, -14026509859/697426329600, 2062196293/54486432000, -3989974979/74724249600, 2439013/42567525, -4596037109/99632332800, 613021979/28021593600, 3033240127/36578304000, -2046617/653837184, 866474507/6276836966400],
[866474507/6276836966400, -679920221/373621248000, 11614474687/1046139494400, -65181504263/1569209241600, 74251645873/697426329600, -492615587/2490808320, 102644956367/373621248000, -25305946559/87178291200, 23409520499/99632332800, -227269902593/1569209241600, 64188105383/1046139494400, 14516634431/201180672000, -1197622087/896690995200],
[-1197622087/896690995200, 1525695617/87178291200, -8800586233/83026944000, 308415783287/784604620800, -77220056327/77491814400, 5304463979/2905943040, -186038426051/74724249600, 26637354127/10378368000, -467089093853/232475443200, 133373184587/112086374400, -183706612697/348713164800, 192413017/1162377216, 132282840127/2414168064000],
[132282840127/2414168064000, -1866476396209/2615348736000, 2040667428953/475517952000, -24757711059413/1569209241600, 27597902895821/697426329600, -31173587791351/435891456000, 5116077905657/53374464000, -42070857451313/435891456000, 50972790156553/697426329600, -64631301332531/1569209241600, 88195348546091/5230697472000, -12555699585959/2615348736000, 2504631949133/2853107712000]]
N=12
N2 = int(N/2)


#Faster function for the calculation of accelerations
masses1 = masses[np.newaxis, :, np.newaxis]
def aGeneral(rAllPlanets):
    deltar = rAllPlanets[np.newaxis, :] - rAllPlanets[:,np.newaxis]
    distanceFactor = (np.sqrt(np.sum(deltar**2, axis=-1))**3)[:,:,np.newaxis]
    return np.sum(np.divide(deltar, distanceFactor, out=np.zeros_like(deltar), where=np.rint(distanceFactor)!=0)*G*masses1, axis=1)



#UI:
# t0 =  373246737.99170226
# t1 = 578499117.7331004
# rinit = np.array(["4144393.343943437", "-4942696.470309006", "-1211826.1813648157"], dtype=np.longdouble)
# vinit = np.array(["11127.632798784025", "7561.947853523697", "7212.976972498207"], dtype=np.longdouble)
# refpos = np.array(["116981803.91779557", "485211077.83533645", "-29756805.910740476"], dtype=np.longdouble)
# target = 6

#testing values:
t0 = 0
t1 = 31536000
rinit = np.array([6478100,0, 0], dtype=np.longdouble)
vinit = np.array([0, 10000, 0], dtype=np.longdouble)
refpos = np.array([0,0,0], dtype=np.longdouble)
target = 3

#r0, v0
import time
starttime = time.time()
#LOADING SELF-CALCULATED EPHEMERIS:
with open("r.npy", "rb") as file:
    rplanets_load = np.load(file)
with open("v.npy", "rb") as file:
    vplanets_load = np.load(file)

print("Time to load files: {:.2f}s".format(time.time() - starttime))

#Startup and Interpolation
planetarydt = dt
dt = 100
sl = int(planetarydt/dt) - 1        #This only works if multiples of 100 (dt) are used as planetarydt
forwards =  sl//2 + sl%2
backwards = sl//2

#redefining sl for later purposes:
sl += 1


T = t1 - t0
steps = int(T/dt) + N2 + 2

rplanets = np.empty([steps,9,3], dtype=np.longdouble)
vplanets = np.empty([steps,9,3], dtype=np.longdouble)



b0 = t0 - dt*N2     #backpoint 0
IDX0 = ceil(b0/planetarydt) + N2old    #used later for slicing
print(IDX0)
IDX1 = int(t1/planetarydt) + N2old
print(IDX1)
p0 = ceil(b0/planetarydt)*sl - round(b0/dt)            #step in terms of dt - nearest calculated point = number of steps to be interpolated backwards & index of the first grid-point to be used
p1 = int(t1/planetarydt)*sl - round(b0/dt)              #from b0 to t1, how many points are used.
print(p0)
print(p1)

rplanets[p0:p1 + 1:sl] = rplanets_load[IDX0:IDX1 + 1]
vplanets[p0:p1 + 1:sl] = vplanets_load[IDX0:IDX1 + 1]

rplanets_load = None
del rplanets_load
vplanets_load = None
del vplanets_load

masses2 = masses[np.newaxis, np.newaxis, :, np.newaxis]
def aGeneral2(rAllPlanets):
    deltar = rAllPlanets[:,np.newaxis, :] - rAllPlanets[:,:,np.newaxis]
    distanceFactor = (np.sqrt(np.sum(deltar**2, axis=-1))**3)[:,:,:,np.newaxis]
    return np.sum(np.divide(deltar, distanceFactor, out=np.zeros_like(deltar), where=np.rint(distanceFactor)!=0)*G*masses1, axis=2)

dt = -dt
for i in range(p0):
    m0 = vplanets[p0 - i]
    k0 = aGeneral(rplanets[p0 - i])
    m1 = vplanets[p0 - i] + k0*dt/2
    k1 = aGeneral(rplanets[p0 - i] + m0*dt/2)
    m2 = vplanets[p0 - i] + k1*dt/2
    k2 = aGeneral(rplanets[p0 - i] + m1*dt/2)
    m3 = vplanets[p0 - i] + k2*dt/2
    k3 = aGeneral(rplanets[p0 - i] + m2*dt/2)
    vplanets[p0 - i - 1] = vplanets[p0 - i] + (k0 + 2*k1 + 2*k2 + k3)*dt/6
    rplanets[p0 - i - 1] = rplanets[p0 - i] + (m0 + 2*m1 + 2*m2 + m3)*dt/6

with tqdm(total=(sl - 1)*4) as pbar:
    for i in range(backwards):
        m0 = vplanets[p0 + sl - i:p1 + 1:sl]
        k0 = aGeneral2(rplanets[p0 + sl - i:p1 + 1:sl])
        pbar.update(1)
        m1 = vplanets[p0 + sl - i:p1 + 1:sl] + k0*dt/2
        k1 = aGeneral2(rplanets[p0 + sl - i:p1 + 1:sl] + m0*dt/2)
        pbar.update(1)
        m2 = vplanets[p0 + sl - i:p1 + 1:sl] + k1*dt/2
        k2 = aGeneral2(rplanets[p0 + sl - i:p1 + 1:sl] + m1*dt/2)
        pbar.update(1)
        m3 = vplanets[p0 + sl - i:p1 + 1:sl] + k2*dt
        k3 = aGeneral2(rplanets[p0 + sl - i:p1 + 1:sl] + m2*dt)
        pbar.update(1)
        vplanets[p0 + sl - i - 1:p1 + 1:sl] = vplanets[p0 + sl - i:p1 + 1:sl] + (k0 + 2*k1 + 2*k2 + k3)*dt/6
        rplanets[p0 + sl - i - 1:p1 + 1:sl] = rplanets[p0 + sl - i:p1 + 1:sl] + (m0 + 2*m1 + 2*m2 + m3)*dt/6

    dt = -dt
    for i in range(forwards):
        m0 = vplanets[p0 + i:p1:sl]
        k0 = aGeneral2(rplanets[p0 + i:p1:sl])      #p1 is the last point that is a multiple of sl. It is not included. This is deliberate behaviour, we will extrapolate later
        pbar.update(1)
        m1 = vplanets[p0 + i:p1:sl] + k0*dt/2
        k1 = aGeneral2(rplanets[p0 + i:p1:sl] + m0*dt/2)
        pbar.update(1)
        m2 = vplanets[p0 + i:p1:sl] + k1*dt/2
        k2 = aGeneral2(rplanets[p0 + i:p1:sl] + m1*dt/2)
        pbar.update(1)
        m3 = vplanets[p0 + i:p1:sl] + k2*dt
        k3 = aGeneral2(rplanets[p0 + i:p1:sl] + m2*dt)
        pbar.update(1)
        vplanets[p0 + i + 1:p1:sl] = vplanets[p0 + i:p1:sl] + (k0 + 2*k1 + 2*k2 + k3)*dt/6
        rplanets[p0 + i + 1:p1:sl] = rplanets[p0 + i:p1:sl] + (m0 + 2*m1 + 2*m2 + m3)*dt/6

print(np.arange(p0 + forwards, p1, sl)[-1])
print(p1)
print(steps)

for i in range(steps - p1 - 1):
    m0 = vplanets[p1 + i]
    k0 = aGeneral(rplanets[p1 + i])
    m1 = vplanets[p1 + i] + k0*dt/2
    k1 = aGeneral(rplanets[p1 + i] + m0*dt/2)
    m2 = vplanets[p1 + i] + k1*dt/2
    k2 = aGeneral(rplanets[p1 + i] + m1*dt/2)
    m3 = vplanets[p1 + i] + k2*dt/2
    k3 = aGeneral(rplanets[p1 + i] + m2*dt/2)
    vplanets[p1 + i + 1] = vplanets[p1 + i] + (k0 + 2*k1 + 2*k2 + k3)*dt/6
    rplanets[p1 + i + 1] = rplanets[p1 + i] + (m0 + 2*m1 + 2*m2 + m3)*dt/6

# from vis import *
# Earth = rplanets[-100:,3]
# lign(Earth, '#3d4782')
# plot()

#Interpolation complete !
# with open("rocketdata.npy", "rb") as file:
#     r = np.load(file)
# print(np.min(mag(r - rplanets[:,target])))
# exit(0)

r = np.empty([steps,3], dtype=np.longdouble)
v = np.empty([steps,3], dtype=np.longdouble)

#DECLARING INITIAL CONDITIONS:
r[N2] = rplanets[N2][3] + rinit     #transformation from earth to other coordinates!
v[N2] = vplanets[N2][3] + vinit


def fs(rpl, rsat):
    deltar = rpl - rsat[np.newaxis ,:]
    distanceFactor = (np.sqrt(np.sum(deltar**2, axis=-1))**3)[:,np.newaxis]
    return np.sum(np.divide(deltar, distanceFactor, out=np.zeros_like(deltar), where=np.rint(distanceFactor)!=0)*(G*masses[:, np.newaxis]), axis=0)

def f(n):
    deltar = rplanets[n] - r[n][np.newaxis ,:]
    distanceFactor = (np.sqrt(np.sum(deltar**2, axis=-1))**3)[:,np.newaxis]
    return np.sum(np.divide(deltar, distanceFactor, out=np.zeros_like(deltar), where=np.rint(distanceFactor)!=0)*(G*masses[:, np.newaxis]), axis=0)



stepli = np.empty([4,2,objcount,3], dtype=np.longdouble)

dt = -dt
for k in range(N2):
    stepli[0, 0] = vplanets[N2 - k]
    stepli[0, 1] = aGeneral(rplanets[N2 - k])
    stepli[1, 0] = vplanets[N2 - k] + stepli[0, 1]*dt/4
    stepli[1, 1] = aGeneral(rplanets[N2 - k] + stepli[0, 0]*dt/4)
    stepli[2, 0] = vplanets[N2 - k] + stepli[1, 1]*dt/4
    stepli[2, 1] = aGeneral(rplanets[N2 - k] + stepli[1, 0]*dt/4)
    stepli[3, 0] = vplanets[N2 - k] + stepli[2, 1]*dt/2
    stepvectm = r[N2 - k] + (stepli[0, 0] + stepli[1, 0]*2 + stepli[2, 0]*2 + stepli[3, 0])*dt/12

    m0 = v[N2 - k]
    k0 = fs(rplanets[N2 - k], r[N2 - k])
    m1 = v[N2 - k] + k0*dt/2
    k1 = fs(stepvectm, m0*dt/2)
    m2 = v[N2 - k] + k1*dt/2
    k2 = fs(stepvectm, m1*dt/2)
    m3 = v[N2 - k] + k2*dt
    k3 = fs(rplanets[N2 - k - 1], m2*dt)
    r[N2 - k - 1] = r[N2 - k] + (m0 + m1*2 + m2*2 + m3)*dt/6
    v[N2 - k - 1] = v[N2 - k] + (k0 + k1*2 + k2*2 + k3)*dt/6

dt = -dt
for k in range(N2):
    stepli[0, 0] = vplanets[N2 + k]
    stepli[0, 1] = aGeneral(rplanets[N2 + k])
    stepli[1, 0] = vplanets[N2 + k] + stepli[0, 1]*dt/4
    stepli[1, 1] = aGeneral(rplanets[N2 + k] + stepli[0, 0]*dt/4)
    stepli[2, 0] = vplanets[N2 + k] + stepli[1, 1]*dt/4
    stepli[2, 1] = aGeneral(rplanets[N2 + k] + stepli[1, 0]*dt/4)
    stepli[3, 0] = vplanets[N2 + k] + stepli[2, 1]*dt/2
    stepvectm = r[N2 + k] + (stepli[0, 0] + stepli[1, 0]*2 + stepli[2, 0]*2 + stepli[3, 0])*dt/12

    m0 = v[N2 + k]
    k0 = fs(rplanets[N2 + k], r[N2 + k])
    m1 = v[N2 + k] + k0*dt/2
    k1 = fs(stepvectm, m0*dt/2)
    m2 = v[N2 + k] + k1*dt/2
    k2 = fs(stepvectm, m1*dt/2)
    m3 = v[N2 + k] + k2*dt
    k3 = fs(rplanets[N2 + k + 1], m2*dt)
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

#Commencing PEC cycle:
n = N
t = N2*dt


corrsum = np.empty([2, 3], dtype=np.longdouble)
with tqdm(total=(steps - 9)) as pbar:
    while t <= T:        #T is defined in general_definitions
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
print("Rocket Trajectory calculated!\nSaving to file...")

with open("rocketdata.npy", "wb") as file:
    np.save(file, r)

print(np.min(mag(r - rplanets[:,target] - refpos[np.newaxis, :])))

# #Further testing the stability of the simulation:

# earthcentricr = r[:500] - rplanets[:500,3]
# earthcentricv = v[:500] - vplanets[:500,3]

# hearth = np.cross(rplanets[0, 3], vplanets[0, 3])
# hsat = np.cross(earthcentricr[0], earthcentricv[0])
# print(np.arccos(np.dot(hearth, hsat)/(mag(hearth)*mag(hsat)))*180/np.pi)

# E = mag(earthcentricv)**2 / 2 - masses[3]*G/mag(earthcentricr)

# x = np.arange(t0, t0 + dt*500, dt, dtype=np.longdouble)
# import matplotlib.pyplot as plt
# plt.scatter(x, E)
# plt.show()


# rplanets = np.flipud(rplanets)
# vplanets = np.flipud(vplanets)
# r0 = r[-N2 - 1]
# v0 = v[-N2 - 1]
# with open("testrocketsim.npz", "wb") as file:
#     np.savez(file, rplanets, vplanets, r0, v0)
