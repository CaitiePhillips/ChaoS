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
QUADS = 2


#OSEM params
s = 12
subsetsMode = 1
K = parameters[0]

iterations = parameters[2]
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

INF_NORM = lambda x: max(x.real, x.imag)
def elNorm(l): 
    return lambda x: int(x.real**l+x.imag**l)
EUCLID_NORM = elNorm(2)

def recon_CT(p, angles, subsetAngles, iterations): 
    lena, mask = imageio.phantom(N, p, True, np.uint32, True)

    #compute m subsets
    subsetsMValues = []
    for subset in subsetAngles:
        mValues = []
        for angle in subset:
            m, inv = farey.toFinite(angle, p)
            mValues.append(m)
        subsetsMValues.append(mValues)

    mt_lena = mojette.transform(lena, angles)
    rt_lena = mojette.toDRT(mt_lena, angles, p, p, p) 
    
    recon, mses, psnrs, ssims = osem_expand(iterations, p, rt_lena, subsetsMValues, finite.frt, finite.ifrt, lena, mask)
    recon = np.abs(recon)

    return recon, np.sqrt(mses), psnrs, ssims


def get_composites(subsetAngles):
    subsetComposites = []
    composites = []
    for subset in subsetAngles:
        compositeSubset = []
        for angle in subset:
            if not farey.is_gauss_prime(angle):
                compositeSubset.append(angle)
                composites.append(angle)
        subsetComposites.append(compositeSubset)
    
    return composites, subsetComposites


def auto_recon_0(p, num_angles, iterations): 
    """
    Creates baseline regular and prime reconstructions to compare future experiment results to.

    Paramaters:
        p: size of image and fractal
        num_angles: number of angles in one quadrant 
        iterations: number of osem iterations 
    """
    p = nt.nearestPrime(N)

    to_plot = {}

    #gauss int recon
    angles, subsetAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles)    
    recon, rmses, psnrs, ssims = recon_CT(p, angles, subsetAngles, iterations)
    to_plot["gaussian integer recon"] = {"rmses":rmses, "psnrs":psnrs, "ssims":ssims}
    path = "CT_results/exp_0/regular_recon_angles_"+ str(angles) + "_its_" + str(iterations) + ".npz"
    np.savez(path, prime=False, angles=angles, recon=recon, rmses=rmses, ssims=ssims, psnrs=psnrs)
      
    #gauss prime recon
    primes, subsetPrimes, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles, prime_only=True)    
    recon, rmses, psnrs, ssims = recon_CT(p, primes, subsetPrimes, iterations)
    to_plot["gaussian prime recon"] = {"rmses":rmses, "psnrs":psnrs, "ssims":ssims}
    path = "CT_results/exp_0/prime_recon_angles_"+ str(angles) + "_its_" + str(iterations) + ".npz"
    np.savez(path, prime=True, angles=primes, recon=recon, rmses=rmses, ssims=ssims, psnrs=psnrs)

    #composite recon
    composites, subsetComposites = get_composites(subsetAngles)
    recon, rmses, psnrs, ssims = recon_CT(p, composites, subsetComposites, iterations)
    to_plot["gaussian int w/o primes recon "] = {"rmses":rmses, "psnrs":psnrs, "ssims":ssims}
    path = "CT_results/exp_0/composite_recon_angles_"+ str(angles) + "_its_" + str(iterations) + ".npz"
    np.savez(path, prime=True, angles=composites, recon=recon, rmses=rmses, ssims=ssims, psnrs=psnrs)

    plot_recons(to_plot)


def plot_recon_0(): 
    fig, (ax_rmse, ax_ssims, ax_psnr) = plt.subplots(1, 3)
    labels = ["gaussian integer recon", "gaussian prime recon", "gaussian integer w/o prime recnon"]
    paths = ["CT_results/exp_0/regular_recon_angles_1000_its_1000.npz", 
            "CT_results/exp_0/prime_recon_angles_1000_its_1000.npz", 
            "CT_results/exp_0/composite_recon_angles_1000_its_1000.npz"]
    for i, path in enumerate(paths): 
        data = np.load(path)
        ax_rmse.plot(data["rmses"])
        ax_ssims.plot(data["ssims"])
        ax_psnr.plot(data["psnrs"], label=labels[i])
        ax_psnr.legend()
    plt.show()
    

def plot_recons(recon_info): 
    fig, (ax_rmse, ax_ssims, ax_psnr) = plt.subplots(1, 3)
    for label, error_info in recon_info.items(): 
        ax_rmse.plot(error_info["rmses"])
        ax_ssims.plot(error_info["ssims"])
        ax_psnr.plot(error_info["psnrs"], label=label)
        ax_psnr.legend()

    plt.show()


def get_subset_index(angle, subsetAngles): 
    """Find subset index of the given angle. 

    Args:
        angle (complex): angle to identify index of
        subsetAngles (list[list[complex]]): list of angles to sort through

    Returns:
        _type_: _description_
    """
    for i, subset in enumerate(subsetAngles): 
        if angle in subset: 
            return i


def auto_recon_1(p, num_angles, iterations):
    """
    complete reconstruction for 
        angles = prime angles + (a, b) + (-a, b) + (b, a) + (-b, a)
    for each set of (a, b)s in the composite list. 


    Args:
        p (int): size of reconstruction and fractal
        num_angles (int): number of angles to use in reconstruction
        iterations (int): number of osem iterations
    """
    to_plot = {}

    angles, subsetAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles)    
    primes, subsetPrimes, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles, prime_only=True)    
    composites, subsetComposites = get_composites(subsetAngles)
    composites = sorted(composites, key=EUCLID_NORM)

    step = 4
    for i in range(0, len(composites), step): 
        #add angles to prime subset
        subsetAngles = list(subsetPrimes)
        angles = list(primes)
        for angle in composites[i:i+step]:
            idx = get_subset_index(angle, subsetComposites)
            subsetAngles[idx].append(angle)
            angles.append(angle)
        #recon with new angle set & save
        recon, rmses, psnrs, ssims = recon_CT(p, angles, subsetAngles, iterations)
        to_plot[str(composites[i:i+step])] = {"composites":composites[i:i+step], "recon":recon, "rmses":rmses, "psnrs":psnrs, "ssims":ssims}
    
    path = "CT_results/exp_1/prime_comp_recons_angles_"+ str(num_angles) + "_its_" + str(iterations) + ".npz"
    np.savez(path, primes=primes, plotInfo=to_plot)
    # plot_recons(to_plot)


def auto_recon_2(p, num_angles, iterations):
    """
    complete reconstruction for 
        angles = prime angles + (a, b) + (-a, b) (almost same as auto_recon_1)
    for each set of (a, b)s in the composite list. 


    Args:
        p (int): size of reconstruction and fractal
        num_angles (int): number of angles to use in reconstruction
        iterations (int): number of osem iterations
    """
    to_plot = {}

    angles, subsetAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles)    
    primes, subsetPrimes, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles, prime_only=True)    
    composites, subsetComposites = get_composites(subsetAngles)
    composites = sorted(composites, key=EUCLID_NORM)

    step = 2
    for i in range(0, len(composites), step): 
        #add angles to prime subset
        subsetAngles = list(subsetPrimes)
        angles = list(primes)
        for angle in composites[i:i+step]:
            idx = get_subset_index(angle, subsetComposites)
            subsetAngles[idx].append(angle)
            angles.append(angle)
        #recon with new angle set & save
        recon, rmses, psnrs, ssims = recon_CT(p, angles, subsetAngles, iterations)
        to_plot[str(composites[i:i+step])] = {"composites":composites[i:i+step], "recon":recon, "rmses":rmses, "psnrs":psnrs, "ssims":ssims}
    
    path = "CT_results/exp_2/prime_comp_recons_angles_"+ str(num_angles) + "_its_" + str(iterations) + ".npz"
    np.savez(path, primes=primes, plotInfo=to_plot)
    # plot_recons(to_plot)
   

def temp_for_now(p, num_angles): 
    angles, subsetAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles)    
    primes, subsetPrimes, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles, prime_only=True)    
    composites, subsetComposites = get_composites(subsetAngles)

    distances = {}
    for angle in composites: 
        min_dist = round(np.abs(np.angle(angle, deg=1) - np.angle(primes[0], deg=1)), 3)

        for prime in primes: 
            dist = np.abs(np.angle(angle, deg=1) - np.angle(prime, deg=1))
            dist = round(dist, 3)
            if dist < min_dist: 
                min_dist = dist

        if dist in distances.keys(): 
            distances[dist].append(angle)
        else:
            distances[dist] = [angle]

    print(distances)




if __name__ == "__main__": 
    p = nt.nearestPrime(N)
    # temp_for_now(p, 20)
    auto_recon_2(p, 20, 2)
# %%
