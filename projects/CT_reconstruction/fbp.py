# -*- coding: utf-8 -*-
"""
Create a finite slice sampling of k-space and reconstruct using MLEM

#No smooth
#Lena, N = 256, 400, Noise, K=2.5, s=4,subset=1, SNR=30
#Phantom, N = 256, 150, Noise, K=2.5, s=4,subset=1, SNR=30
#Cameraman, N = 256, 500, Noise, K=2.5, s=4,subset=1, SNR=30

#Reduction Factors
#0.5:
#Phantom: N=256, i=381, s=30, h=8, K=1.2, k=1;
#0.25:
#Phantom: N=256, i=761, s=12, h=12, K=0.4, k=1;
#0.125:
#Phantom: N=256, i=761, s=6, h=12, K=0.15, k=1;

Copyright 2018 Shekhar S. Chandra

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# %%
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
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time
import matplotlib
import os
import math

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

# helper functions -------------------------------------------------------------
def compute_slopes(subsetsAngles, twoQuads , p): 
        """
        Compute gradients of given angles. 

        subsetsAngles (ndarray(dtype=float, ndim=?)): array of angles
        twoQuads (bool): one or two quadrants 
        """
        subsetsMValues = []
        for angles in subsetsAngles:
            mValues = []
            for angle in angles:
                m, inv = farey.toFinite(angle, p)
                mValues.append(m)
                #second quadrant
                if twoQuads:
                    if m != 0 and m != p: #dont repeat these
                        m = p-m
                        mValues.append(m)
            subsetsMValues.append(mValues)
        return subsetsMValues

def fill_dft(rt, subsetsMValues, powSpectLena): 
    """
    fill 2D FT space with projections via FST

    subsetsMValues (ndarray(dtype:float, dim:?)): array of prjoection angle gradients
    powSpectLena (ndarray(dtype:float, dim:?)): empty 2D FT Space to be filled
    """
    mValues = [m for mSet in subsetsMValues for m in mSet] 
    for m in mValues: 
        slice = fftpack.fft(rt[m])
        radon.setSlice(m, powSpectLena, slice)

    powSpectLena = fftpack.fftshift(powSpectLena)
    # powSpectLena = np.abs(powSpectLena)

    powSpectLena = powSpectLena + np.flipud(powSpectLena) # conj symmetric

def plot_angles(angles): 
    real = [angle.real for angle in angles]
    im = [angle.imag for angle in angles]
    plt.plot(real, im, 'o')

    angles = [np.pi * i / 6 for i in range(0, 7)]
    x = [7 * np.cos(angle) for angle in angles]
    y = [7 * np.sin(angle) for angle in angles]
    for i in range(len(x)):
        plt.plot([0, x[i]], [0, y[i]], 'r-')

    plt.xlim([-7, 7])
    plt.ylim([-7, 7])
    plt.grid(visible=True, which='both', axis='both')
    plt.show()

parameters = [0.4, 1, 760, 12, 12.0] #r=4

#parameters
n = 256
k = parameters[1]
M = int(k*n)
N = n 
K = parameters[0]
s = parameters[3]
iterations = 100 #parameters[2]
subsetsMode = 1
SNR = 20
floatType = float# np.complex
twoQuads = True
addNoise = True
plotCroppedImages = True
plotColourBar = True
plotIncrement = 2
smoothReconMode = 2 #0-None,1-TV,2-NL,3-Median
smoothIncrement = 10
smoothMaxIteration = iterations/2
relaxIterationFactor = int(0.01*iterations)
#smoothMaxIteration2 = iterations-1
smoothMaxIteration2 = iterations-relaxIterationFactor*smoothIncrement
print("N:", N, "M:", M, "s:", s, "i:", iterations)

pDash = nt.nearestPrime(N)
print("p':", pDash)

iterations = 100
addNoise = False
max_angles = 56


angles, subsetsAngles, _ = mojette.angleSubSets_Symmetric(s,subsetsMode,N,N,8,True,K)

# perpAngle = farey.farey(1,0)
# angles.append(perpAngle)
# subsetsAngles[0].append(perpAngle)

# plot_angles(angles)
# opposite_angles = []
# for angleSet in subsetsAngles: 
#     new = [farey.farey(-1* angle.imag, 1 * angle.real) for angle in angleSet]
#     opposite_angles.append(new)
#     angles += new 

# perpAngle = farey.farey(1,0)
# angles.append(perpAngle)
# subsetsAngles[0].append(perpAngle)
# plot_angles(angles)


print("Number of Angles:", len(angles))
print("angles:", angles)

p = nt.nearestPrime(M)
lena, mask = imageio.phantom(N, p, True, np.uint32, True)

# %%
# #-------------------------------

#acquired Mojette projections
mt_lena = mojette.transform(lena, angles)
#convert to radon projections for recon
rt_lena = mojette.toDRT(mt_lena, angles, p, N, N) 
recon = finite.ifrt(rt_lena, p)

# %% Look at diff between projs in 2dft and rt projs 
fftLena = fftpack.fft2(lena) #the '2' is important
fftLenaShifted = fftpack.fftshift(fftLena)

# %%
ramp_up = np.linspace(0, 257, 128)
ramp_down = np.linspace(257, 0, 129)
ramp = np.concatenate((ramp_down, ramp_up))

recon = finite.ifrt(rt_lena, p)

for i, proj in enumerate(rt_lena): 
    rt_lena[i][:] = fftpack.ifft(fftpack.fft(proj) * ramp)

ramp_recon = finite.ifrt(rt_lena, p)

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(recon)
ax2.imshow(ramp_recon)
ax3.imshow(abs(recon - ramp_recon))
plt.show()

    
# %%
