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

def osem_expand(iterations, p, g_j, os_mValues, projector, backprojector,
                         image, mask, epsilon=1e3, dtype=np.int32):
    '''
    # Gary's implementation
    # From Lalush and Wernick;
    # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
    # where g = \sum (h f^\hat)                                   ... (**)
    #
    # self.f is the current estimate f^\hat
    # The following g from (**) is equivalent to g = \sum (h f^\hat)
    '''
    norm = False
    center = False
    fdtype = floatType
    f = np.ones((p,p), fdtype)
    
    mses = []
    psnrs = []
    ssims = []
    for i in range(0, iterations):
        for j, mValues in enumerate(os_mValues):
#            print("Subset:", j)
            muFinite = len(mValues)
            
            g = projector(f, p, fdtype, mValues)
        
            # form parenthesised term (g_j / g) from (*)
            r = np.copy(g_j)
            for m in mValues:
                for y in range(p):
                    r[m,y] /= g[m,y]
        
            # backproject to form \sum h * (g_j / g)
            g_r = backprojector(r, p, norm, center, 1, 0, mValues) / muFinite
        
            # Renormalise backprojected term / \sum h)
            # Normalise the individual pixels in the reconstruction
            f *= g_r
        
        if smoothReconMode > 0 and i % smoothIncrement == 0 and i > 0: #smooth to stem growth of noise
            if smoothReconMode == 1:
                print("Smooth TV")
                f = denoise_tv_chambolle(f, 0.02, multichannel=False)
            elif smoothReconMode == 2:
                h = parameters[4]
                if i > smoothMaxIteration:
                    h /= 2.0
                if i > smoothMaxIteration2:
                    h /= 4.0
                print("Smooth NL h:",h)
                fReal = denoise_nl_means(np.real(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                # fImag = denoise_nl_means(np.imag(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                f = fReal #+1j*fImag
            elif smoothReconMode == 3:
                print("Smooth Median")
                fReal = ndimage.median_filter(np.real(f), 3)
                # fImag = ndimage.median_filter(np.imag(f), 3)
            f = fReal #+1j*fImag
            
        if i%plotIncrement == 0:
            img = imageio.immask(image, mask, N, N)
            recon = imageio.immask(f, mask, N, N)
            recon = np.abs(recon)
            mse = imageio.immse(img, recon)
            psnr = imageio.impsnr(img, recon)
            ssim = imageio.imssim(img.astype(float), recon.astype(float))
            print("RMSE:", math.sqrt(mse), "PSNR:", psnr, "SSIM:", ssim)
            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
        
    return f, mses, psnrs, ssims

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

#parameter sets (K, k, i, s, h)
#phantom
#parameters = [1.2, 1, 381, 30, 8.0] #r=2
parameters = [0.4, 1, 100, 12, 12.0] #r=4
#cameraman
#parameters = [1.2, 1, 381, 30, 8.0] #r=2

#parameters
n = 256 
k = parameters[1]
M = int(k*n)
N = n 
K = parameters[0]
s = parameters[3]
iterations = parameters[2]
subsetsMode = 1
SNR = 30
floatType = np.float
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
angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,N,N,2,True,K, max_angles=20, prime_only=True)
print("Number of Angles:", len(angles))
print("angles:", angles)
ct_angles = angles

p = nt.nearestPrime(M)
print("p:", p)

#check if Katz compliant
if not mojette.isKatzCriterion(N, N, angles):
    print("Warning: Katz Criterion not met")

#create test image
lena, mask = imageio.phantom(N, p, True, np.uint32, True)

#compute m subsets
subsetsMValues = []
for angles in subsetsAngles:
    mValues = []
    for angle in angles:
        m, inv = farey.toFinite(angle, p)
        mValues.append(m)
    subsetsMValues.append(mValues)
# print(subsetsMValues)

#CT
mt_lena = mojette.transform(lena, ct_angles)
rt_lena = mojette.toDRT(mt_lena, ct_angles, p, p, p) 

recon, mses, psnrs, ssims = osem_expand(iterations, p, rt_lena, subsetsMValues, finite.frt, finite.ifrt, lena, mask)
recon = np.abs(recon)

mse = imageio.immse(imageio.immask(lena, mask, N, N), imageio.immask(recon, mask, N, N))
ssim = imageio.imssim(imageio.immask(lena, mask, N, N).astype(float), imageio.immask(recon, mask, N, N).astype(float))
psnr = imageio.impsnr(imageio.immask(lena, mask, N, N), imageio.immask(recon, mask, N, N))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)


reconLena = finite.ifrt(rt_lena, p)
diff = lena - recon
mseValues = np.array(mses)
psnrValues = np.array(psnrs)
ssimValues = np.array(ssims)

if plotCroppedImages:
    print(lena.shape)
    print(mask.shape)
    lena = imageio.immask(lena, mask, N, N)
    reconLena = imageio.immask(reconLena, mask, N, N)
    # reconNoise = imageio.immask(reconNoise, mask, N, N)
    recon = imageio.immask(recon, mask, N, N)
    diff = imageio.immask(diff, mask, N, N)

fig, ax = plt.subplots(figsize=(24, 9))
plt.subplot(121)
rax = plt.imshow(reconLena, interpolation='nearest', cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray', orientation="horizontal")
plt.title('Inverse DRT')
plt.subplot(122)
rax2 = plt.imshow(recon, interpolation='nearest', cmap='gray')
rcbar2 = plt.colorbar(rax2, cmap='gray', orientation="horizontal")
plt.title('Reconstruction')


fig, ax = plt.subplots(figsize=(24, 9))
plt.subplot(221)
rax = plt.imshow(lena, interpolation='nearest', cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Original Image')
plt.subplot(222)
rax = plt.imshow(reconLena, interpolation='nearest', cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Inverse DRT')
plt.subplot(223)
rax2 = plt.imshow(recon, interpolation='nearest', cmap='gray')
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('Reconstruction')
plt.subplot(224)
rax3 = plt.imshow(diff, interpolation='nearest', cmap='gray')
rcbar3 = plt.colorbar(rax3, cmap='gray')
plt.title('Reconstruction Errors')
# plt.subplot(153)
# rax = plt.imshow(reconLena - lena, interpolation='nearest', cmap='gray')
# rcbar = plt.colorbar(rax, cmap='gray')
# plt.title('Noise')

fig, ax = plt.subplots(figsize=(24, 9))


incX = np.arange(0, len(mses))*plotIncrement

plt.subplot(131)
plt.plot(incX, np.sqrt(mseValues))
plt.title('Error Convergence of the Finite OSEM')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.subplot(132)
plt.plot(incX, psnrValues)
plt.ylim(0, 45.0)
plt.title('PSNR Convergence of the Finite OSSEM')
plt.xlabel('Iterations')
plt.ylabel('PSNR')
plt.subplot(133)
plt.plot(incX, ssimValues)
plt.ylim(0, 1.0)
plt.title('Simarlity Convergence of the Finite OSSEM')
plt.xlabel('Iterations')
plt.ylabel('SSIM')

plt.show()