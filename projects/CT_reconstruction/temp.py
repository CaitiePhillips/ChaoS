from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import finitetransform.mojette as mojette
import finitetransform.radon as radon
import finitetransform.imageio as imageio #local module
import finitetransform.farey as farey #local module
import finitetransform.numbertheory as nt #local modules
import finite
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means
import scipy.fftpack as fftpack
import pyfftw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time
import matplotlib
import os
import math
matplotlib.use('Qt4Agg')

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

parameters = [0.4, 1, 760, 12, 12.0] #r=4
#cameraman
#parameters = [1.2, 1, 381, 30, 8.0] #r=2

#parameters
n = 4
k = parameters[1]
M = int(k*n)
N = n 
K = parameters[0]
s = parameters[3]
subsetsMode = 1
floatType = np.complex
twoQuads = True
max_angles = 1


angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,N,N,1,True,K, max_angles = max_angles)
#angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,M,M,1,True,K)
perpAngle = farey.farey(1,0)
angles.append(perpAngle)
subsetsAngles[0].append(perpAngle)
print("angles:", angles)

p = nt.nearestPrime(M)
print("p:", p)

lena, mask = imageio.phantom(N, p, True, np.uint32, True)
mt_lena = mojette.transform(lena, angles)

size = int(N + N/2)
dyadic = True
if N % 2 == 1: # if odd, assume prime
    size = int(N+1)
    dyadic = False
    
m = 0

frtSpace = np.zeros( (size,N) )

if dyadic:
    print("Dyadic size")
    for index, proj in enumerate(mt_lena):
        p, q = farey.get_pq(angles[index])

        m, inv = farey.toFinite(angles[index], N)
        frtSpace[m][:] = mojette.finiteProjection(proj, angles[index], N, N, N, False)

print(frtSpace)
print("end")