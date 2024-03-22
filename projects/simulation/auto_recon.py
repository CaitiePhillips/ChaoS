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
from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import finitetransform.mojette as mojette
import finitetransform.radon as radon
import finitetransform.imageio as imageio #local module
import finitetransform.farey as farey #local module
import finitetransform.numbertheory as nt #local modules
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means
import scipy.fftpack as fftpack
import pyfftw
import numpy as np
import finite
import time
import math

import matplotlib
matplotlib.use('Qt4Agg')
from scipy import ndimage


# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

#parameter sets (K, k, i, s, h)
#phantom
#parameters = [1.2, 1, 381, 30, 8.0] #r=2
parameters = [0.4, 1, 300, 12, 12.0] #r=4
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
floatType = np.complex
twoQuads = True
addNoise = False
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

INF_NORM = lambda x: max(x.real, x.imag)
EUCLID_NORM = lambda x: x.real**2+x.imag**2
def elNorm(l): 
    return lambda x: x.real**l+x.imag**l

def osem_expand_complex(iterations, p, g_j, os_mValues, projector, backprojector, image, mask, epsilon=1e3, dtype=np.int32):
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
#                r[m,:] = g_j[m,:] / g[m,:]
                for y in range(p):
                    r[m,y] /= g[m,y]
        
            # backproject to form \sum h * (g_j / g)
            g_r = backprojector(r, p, norm, center, 1, 0, mValues) / muFinite
        
            # Renormalise backprojected term / \sum h)
            # Normalise the individual pixels in the reconstruction
            f *= g_r
        
        if smoothReconMode > 0 and i % smoothIncrement == 0 and i > 0: #smooth to stem growth of noise
            if smoothReconMode == 1:
                f = denoise_tv_chambolle(f, 0.02, multichannel=False)
            elif smoothReconMode == 2:
                h = parameters[4]
                if i > smoothMaxIteration:
                    h /= 2.0
                if i > smoothMaxIteration2:
                    h /= 4.0
                fReal = denoise_nl_means(np.real(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                fImag = denoise_nl_means(np.imag(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                f = fReal +1j*fImag
            elif smoothReconMode == 3:
                fReal = ndimage.median_filter(np.real(f), 3)
                fImag = ndimage.median_filter(np.imag(f), 3)
            f = fReal +1j*fImag
            
        if i%plotIncrement == 0:
            img = imageio.immask(image, mask, N, N)
            recon = imageio.immask(f, mask, N, N)
            recon = np.abs(recon)
            mse = imageio.immse(img, recon)
            psnr = imageio.impsnr(img, recon)
            ssim = imageio.imssim(img.astype(float), recon.astype(float))
            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)

        print("iteration:", str(i) + "/" + str(iterations), end = "\r")
        
    return f, mses, psnrs, ssims


test_angles = [15, 25, 50, 75, 100]
primes = [0, 1]
rmse_all = [[], []]
ssim_all = [[], []]
psnr_all = [[], []]

for j, prime in enumerate(primes): 
    for k, max_angles in enumerate(test_angles):
        angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,N,N,1,True,K, prime_only=prime, max_angles=max_angles, norm=elNorm(2))
        #angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,M,M,1,True,K)
        perpAngle = farey.farey(1,0)
        angles.append(perpAngle)
        subsetsAngles[0].append(perpAngle)

        p = nt.nearestPrime(M)

        #create test image
        #lena, mask = imageio.lena(N, p, True, np.uint32, True)
        lena, mask = imageio.phantom(N, p, True, np.uint32, True)
        #lena, mask = imageio.cameraman(N, p, True, np.uint32, True)

        #-------------------------------
        #k-space
        #2D FFT
        fftLena = fftpack.fft2(lena) #the '2' is important
        fftLenaShifted = fftpack.fftshift(fftLena)
        #power spectrum
        powSpectLena = np.abs(fftLenaShifted)

        #add noise to kSpace
        noise = finite.noise(fftLenaShifted, SNR)
        if addNoise:
            fftLenaShifted += noise

        #Recover full image with noise
        reconLena = fftpack.ifft2(fftLenaShifted) #the '2' is important
        reconLena = np.abs(reconLena)
        reconNoise = lena - reconLena

        mse = imageio.immse(lena, np.abs(reconLena))
        ssim = imageio.imssim(lena.astype(float), np.abs(reconLena).astype(float))
        psnr = imageio.impsnr(lena, np.abs(reconLena))

        #compute lines
        centered = True
        subsetsLines = []
        subsetsMValues = []
        mu = 0
        for angles in subsetsAngles:
            lines = []
            mValues = []
            for angle in angles:
                m, inv = farey.toFinite(angle, p)
                u, v = radon.getSliceCoordinates2(m, powSpectLena, centered, p)
                lines.append((u,v))
                mValues.append(m)
                #second quadrant
                if twoQuads:
                    if m != 0 and m != p: #dont repeat these
                        m = p-m
                        u, v = radon.getSliceCoordinates2(m, powSpectLena, centered, p)
                        lines.append((u,v))
                        mValues.append(m)
            subsetsLines.append(lines)
            subsetsMValues.append(mValues)
            mu += len(lines)

        #samples used
        sampleNumber = (p-1)*mu

        #-------------
        # Measure finite slice

        drtSpace = np.zeros((p+1, p), floatType)
        for lines, mValues in zip(subsetsLines, subsetsMValues):
            for i, line in enumerate(lines):
                u, v = line
                sliceReal = ndimage.map_coordinates(np.real(fftLenaShifted), [u,v])
                sliceImag = ndimage.map_coordinates(np.imag(fftLenaShifted), [u,v])
                slice = sliceReal+1j*sliceImag
            #    print("slice", i, ":", slice)
                finiteProjection = fftpack.ifft(slice) # recover projection using slice theorem
                drtSpace[mValues[i],:] = finiteProjection
        #print("drtSpace:", drtSpace)

        
        recon, mses, psnrs, ssims = osem_expand_complex(iterations, p, drtSpace, subsetsMValues, finite.frt_complex, finite.ifrt_complex, lena, mask)
        rmse_all[prime].append(np.array(mses))
        psnr_all[prime].append(np.array(psnrs))
        ssim_all[prime].append(np.array(ssims))
        print("\n" + (j + 1) * k + "/" + str(len(primes) * len(test_angles)))
        
        recon = np.abs(recon)

        mse = imageio.immse(imageio.immask(lena, mask, N, N), imageio.immask(recon, mask, N, N))
        ssim = imageio.imssim(imageio.immask(lena, mask, N, N).astype(float), imageio.immask(recon, mask, N, N).astype(float))
        psnr = imageio.impsnr(imageio.immask(lena, mask, N, N), imageio.immask(recon, mask, N, N))
        print("max angles", max_angles)
        print("RMSE:", math.sqrt(mse))
        print("SSIM:", ssim)
        print("PSNR:", psnr)
        print()

path = "resuts/errors/recon_its_" + str(iterations) + ".npz"
np.savez("resuts/errors/recon_its", angles=test_angles, rmse=rmse_all, psnr=psnr_all, ssim=ssim_all)


from matplotlib import pyplot as plt
for prime in range(2):
    for i, rmse in enumerate(rmse_all[prime]): 
        plt.plot(rmse, label="angles: " + str(test_angles[i]) + " prime: " + str(bool(prime)))
        
plt.title("RMSE")
plt.legend()

plt.figure()
from matplotlib import pyplot as plt
for prime in range(2):
    for i, psnr in enumerate(psnr_all[prime]): 
        plt.plot(psnr, label="angles: " + str(test_angles[i]) + " prime: " + str(bool(prime)))
        
plt.title("PSNR")
plt.legend()

plt.figure()
from matplotlib import pyplot as plt
for prime in range(2):
    for i, ssim in enumerate(ssim_all[prime]): 
        plt.plot(ssim, label="angles: " + str(test_angles[i]) + " prime: " + str(bool(prime)))
plt.title("SSIM")
plt.legend()


plt.show()

