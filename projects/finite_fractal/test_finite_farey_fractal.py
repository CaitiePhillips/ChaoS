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
import random

import scipy.fftpack as fftpack
import pyfftw
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft
# %%
#parameters
N = 256
M = 1 * N
K = 1
twoQuads = True
print("N:", N, "M:", M)
p = nt.nearestPrime(M) 
print("p:", p)

INF_NORM = lambda x: max(x.real, x.imag)
# RAND_NORM = lambda x: x.real**2+x.imag**2
def elNorm(l): 
    return lambda x: x.real**l+x.imag**l

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

def createFractal(lines, p, plot=True, ax=plt, title="Fractal"): 
    maxLines = len(lines)   
    color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
    image = np.zeros((p,p))

    for i, line in enumerate(lines):
        u, v = line
        c=next(color)
        if plot: 
            ax.plot(u, v, '.', markersize=1, c=c)

        image[u,v] = 1

        if i == maxLines:
            break 

    if plot: 
        try:
            ax.set_title(title)
        except AttributeError:
            ax.title(title)
    return image

def addCentreTile(image, radius): 
    for i, row in enumerate(image): 
        for j, cell in enumerate(row): 
            if (i - 128) **2 + (j - 128) **2 < radius ** 2:
                image[i][j] = 1

def plotFractal(angles, plotReg=True, plotColour=True, save=False, ax=plt, title="fractal"): 
    (lines, mValues) = calcFiniteLines(angles)
    fractal = createFractal(lines, p, plot=plotColour, ax=ax, title=title)
    if plotReg: 
        if plotColour: 
            plt.figure()
        plt.imshow(fractal)
    if save: 
        path = "results/" + title.replace(" ", "_") + "npz"
        np.savez(path, fractal=fractal)

# %% get projection angles
fareyAngles, fareyLengths = mojette.angleSet_Symmetric(p, p, 1,True,K, prime_only=False, max_angles=20, norm=elNorm(2))
gaussAngles, gaussLengths = mojette.angleSet_Symmetric(p, p, 1,True,K, prime_only=True, max_angles=20, norm=elNorm(2))
perpAngle = farey.farey(1,0)
fareyAngles.append(perpAngle)
 

plotFractal(fareyAngles, plotColour=False)
plt.figure()
plotFractal(gaussAngles, plotColour=False)

#angles in fareyAngles but not in gaussAngles (like composite numbers)
# $$fareyNoGauss = \{\theta; \theta \in fareyAngles, \theta \notin gaussAngles\}     $$
fareyNoGauss = []
for angle in fareyAngles: 
    if angle not in gaussAngles: 
        fareyNoGauss.append(angle)

plt.figure()
plotFractal(fareyNoGauss, plotColour=False)

#angles in gaussAngles but not in fareyAngles (prime angles additional to the farey angles)
# $$gaussNoFarey = \{\theta; \theta \in gaussAngles, \theta \notin fareyAngles\}     $$
gaussNoFarey = []
for angle in gaussAngles: 
    if angle not in fareyAngles: 
        gaussNoFarey.append(angle)

plt.figure()
plotFractal(gaussNoFarey, plotColour=True)     

plt.show()