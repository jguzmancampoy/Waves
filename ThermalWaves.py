#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate



def conv(x):
    File = open(x,"r")
    lines = [line.split() for line in File]
    del lines[-1]
    print(lines)
    a = [];b = []
    for i in lines:
        a.append(float(i[0]))
        b.append(float(i[1]))
    return(a,b)


time60 = conv('60sec.txt')[0]
temp60 = conv('60sec.txt')[1]
time240 = conv('240sec.txt')[0]
temp240 = conv('240sec.txt')[1]
time120 = conv('120seconds.txt')[0]
temp120 = conv('120seconds.txt')[1]
time480 = conv('480sec.txt')[0]
temp480 = conv('480sec.txt')[1]


#plt.plot(time120,temp120)
#plt.xlabel("Time Steps")
#plt.ylabel("Temprature")
#plt.title("120 Seconds")
#plt.grid()
#plt.show()




def integrate(time,temp,period):
    timesec = [i/10 for i in time]
    datacos = [temp[i]*(np.cos(2*np.pi*timesec[i]/period)) for i in range(len(timesec))]
    datasin = [temp[i]*(np.sin(2*np.pi*timesec[i]/period)) for i in range(len(timesec))]

    an =(scipy.integrate.simps(datacos,timesec)/(period*2))
    bn = (scipy.integrate.simps(datasin,timesec)/(period*2))
    
    trans = np.sqrt(an**2 + bn**2)
    if period == 60:
        phase = np.arctan(an/bn)+2*np.pi
    else: 
        phase = -np.arctan(an/bn) + np.pi
    dtrans = (2*np.pi/period)*(7.975e-3**2)/(2*(np.log(trans/63.7))**2)
    dphase = (2*np.pi/period)*(7.975e-3**2)/(2*(phase)**2)
    
    """Error Propagation Simpson's Rule"""
    
    erroran = abs(1/90 * (0.1)**5 * max(temp))
    deltabeta = ((an+bn)/(((an**2) +(bn**2))**(0.5)))* erroran
    
    return [dtrans,dphase,deltabeta,erroran]



fouriertrans = [integrate(time60,temp60,60)[0],integrate(time120, temp120,120)[0],
         integrate(time240, temp240,240)[0],integrate(time480,temp480,480)[0]]
fourierphase = [integrate(time60,temp60,60)[1],integrate(time120, temp120,120)[1],
         integrate(time240, temp240,240)[1],integrate(time480,temp480,480)[1]]
time = [1.,2.,4.,8.]


phystrans = [1.835e-7,2.17e-7,3.96e-7,8.207e-7]
physphase = [1.125e-7,1.172e-7,1.004e-7,1.285e-7]


def linfit(x,a,b):
    return (a*x +b)
    
popt1, pcov1 = curve_fit(linfit, time, phystrans)
popt2, pcov2 = curve_fit(linfit, time, physphase)
popt3, pcov3 = curve_fit(linfit, time, fouriertrans)
popt4, pcov4 = curve_fit(linfit, time, fourierphase)
#print([popt1[1],popt2[1],popt3[1],popt4[1]])


def errortrans(x,period):
    ohm = np.pi/period
    error = (7.975e-3**2*ohm*((np.log(x/63.7))**-3))*0.5
    return error


"""
errorphystrans = [errortrans(i,120) for i in phystrans]
print(errorphystrans)
plt.plot(time,fouriertrans,label = "Fourier Trans")
plt.plot(time,fourierphase, label = "Fourier Phase")
plt.errorbar(time,phystrans,yerr = errorphystrans,label = "Physical Trans")
plt.errorbar(time,physphase,yerr = errorphystrans, label = "Physical Phase")


plt.legend()
plt.grid()
plt.ylim(0,9e-7)
plt.xlabel("Time (min)")
plt.ylabel("D  (m^2/s)")

plt.show()
"""
"""

freq = [2.09,4.21,6.32,8.44,10.52,12.57,14.61,16.65,18.61,20.60,22.53,24.40,26.24,
        28.03,29.84,31.53,33.20,34.79,36.33,37.85,39.24,40.64,41.90,43.11,44.24,
        45.35,46.38,47.32,48.16,48.92,49.63,50.25,50.82,51.31,51.68,51.99,52.29,52.55]
n = np.arange(1.,len(freq)+1.)
om = [2*np.pi*f for f in freq]
lam = (80/n)
wavevector = 2*np.pi/lam

plt.scatter(wavevector,om)
plt.show()

freq_tr = []
"""
"""
Transmission factor versus frequency
"""
"""


vgroup = freq/wavevector

newfreq = freq
freq2= []
delom = []

for i in range(37):
    freq2.append(freq[i+1]-freq[i])

#freq3.remove(freq3[-1]) 

vphase =[(80/(2*np.pi)) * f for f in freq2] 

plt.scatter(freq[:37],vphase, c ='g')
plt.scatter(freq,vgroup,c ='b')
plt.xlabel("Frequency(Hz)")
plt.ylabel("Velocity")
plt.title("Phase and group velocities")
plt.show()
    
freq = [15.08,24.03,34.94,45.01,50.75,51.95,52.32,52.45,52.56]
volts = [5.12,5.04,4.88,4.,3.28,2.48,1.36,0.88,0.48]
trrat = [i/5.4 for i in volts]
plt.scatter(freq,trrat)
plt.grid()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Transmission Ratio")
plt.show()
"""

"""I am sane, I promise

          _
             ,.-" "-.,
            /   ===   \
           /  =======  \
        __|  (o)   (0)  |__      
       / _|    .---.    |_ \         
      | /.----/ O O \----.\ |       
       \/     |     |     \/        
       |                   |            
       |                   |           
       |                   |          
       _\   -.,_____,.-   /_         
   ,.-"  "-.,_________,.-"  "-.,
  /          |       |          \  
 |           l.     .l           | 
 |            |     |            |
 l.           |     |           .l             
  |           l.   .l           | \,     
  l.           |   |           .l   \,    
   |           |   |           |      \,  
   l.          |   |          .l        |
    |          |   |          |         |
    |          |---|          |         |
    |          |   |          |         |
    /"-.,__,.-"\   /"-.,__,.-"\"-.,_,.-"\
   |            \ /            |         |
   |             |             |         |
    \__|__|__|__/ \__|__|__|__/ \_|__|__/ 
    
"""