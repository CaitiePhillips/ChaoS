# -*- coding: utf-8 -*-
"""
Create the Farey FInite Fractal as a sampling pattern for MRI

All figures and code pertaining to the display, saving and generation of fractals, 
are covered under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International Public License: http://creativecommons.org/licenses/by-nc-sa/4.0/.
For publication and commercial use of this content, please obtain a suitable license 
from Shekhar S. Chandra.
"""
# %%
from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import finitetransform.numbertheory as nt #local modules
import finitetransform.mojette as mojette
import finitetransform.radon as radon
import finitetransform.imageio as imageio #local module
import finitetransform.farey as farey #local module
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

import scipy.fftpack as fftpack
import pyfftw
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

#parameters
N = nt.nearestPrime(256)
M = 2 * N
K = 1
twoQuads = True
print("N:", N, "M:", M)
p = nt.nearestPrime(M)
print("p:", p)
pDash = nt.nearestPrime(N)
print("p':", pDash)
#angles = mojette.angleSet_Finite(pDash, 2)


def calcFiniteLines(angles): 
    powerSpect = np.zeros((p,p))
    centered = True
    lines = []
    mValues = []

    for angle in angles:
        m, inv = farey.toFinite(angle, p)
        u, v = radon.getSliceCoordinates2(m, powerSpect, centered, p)
        lines.append((u,v))
        mValues.append(m)
        #second quadrant
        if twoQuads:
            if m != 0 and m != p: #dont repeat these
                m = p-m
                u, v = radon.getSliceCoordinates2(m, powerSpect, centered, p)
                lines.append((u,v))
                mValues.append(m)
    
    return(lines, mValues)

def createFractal(lines, ax, p, plot=True): 
    maxLines = len(lines)   
    color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
    image = np.zeros((p,p))

    for i, line in enumerate(lines):
        u, v = line
        c=next(color)
        if plot: 
            ax.plot(u, v, '.', markersize=1, c=c)

        image[u,v] = 255

        if i == maxLines:
            break   
    return image

INF_NORM = lambda x: max(x.real, x.imag)
EUCLID_NORM = lambda x: x.real**2+x.imag**2
def elNorm(l): 
    return lambda x: x.real**l+x.imag**l

fareyAngles, fareyLengths = mojette.angleSet_Symmetric(N,N,1,True,K, prime_only=False, max_angles=20, norm=elNorm(1))
gaussAngles, gaussLengths = mojette.angleSet_Symmetric(N,N,1,True,K, prime_only=True, max_angles=20, norm=elNorm(1))
perpAngle = farey.farey(1,0)
fareyAngles.append(perpAngle)
gaussAngles.append(perpAngle)

#angles in fareyAngles but not in gaussAngles (like composite numbers)
fareyNoGauss = []
for angle in fareyAngles: 
    if angle not in gaussAngles: 
        fareyNoGauss.append(angle)

#angles in gaussAngles but not in fareyAngles (prime angles additional to the farey angles)
gaussNoFarey = []
for angle in gaussAngles: 
    if angle not in fareyAngles: 
        gaussNoFarey.append(angle)

(fareyLines, fareyMValues) = calcFiniteLines(fareyAngles)
(gaussLines, gaussMValues) = calcFiniteLines(gaussAngles)
(fareyNoGaussLines, fareyNoGaussMValues) = calcFiniteLines(fareyNoGauss)
(gaussNoFareyLines, gaussNoFareyMValues) = calcFiniteLines(gaussNoFarey)

plotColour = False

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), squeeze=True)
ax = np.ravel(ax)
fareyImage = createFractal(fareyLines, ax[0], p, plot=plotColour)
gaussImage = createFractal(gaussLines, ax[1], p, plot=plotColour)
ax[0].set_title("Farey Fractal")
ax[1].set_title("Gauss Fractal")

if plotColour: #if plotColour, will need extra figure, else, prev fig, ax will be used
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), squeeze=True)
ax[0].imshow(fareyImage)
ax[1].imshow(gaussImage)
ax[0].set_title("Farey Fractal")
ax[1].set_title("Gauss Fractal")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), squeeze=True)
ax = np.ravel(ax)
fareyNoGauss = createFractal(fareyNoGaussLines, ax[0], p, plot=plotColour)
gaussNoFarey = createFractal(gaussNoFareyLines, ax[1], p, plot=plotColour)
ax[0].set_title("Farey not in Gauss")
ax[1].set_title("Gauss not in Farey")

missingFarey = np.maximum(fareyImage - gaussImage, np.zeros((p, p)))
missingGauss = np.maximum(gaussImage - fareyImage, np.zeros((p, p)))
if plotColour: #if plotColour, will need extra figure, else, prev fig, ax will be used
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), squeeze=True)
ax[0].imshow(missingFarey)
ax[1].imshow(missingGauss)
ax[0].set_title("Farey not in Gauss")
ax[1].set_title("Gauss not in Farey")

plt.show()

# # imageio.imsave("farey_image_"+str(p)+"_"+str(K)+".png", fareyImage)
