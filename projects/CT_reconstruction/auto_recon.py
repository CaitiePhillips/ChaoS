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
from scipy import ndimage
import scipy
import pyfftw
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time
import matplotlib
import os
import math
import random

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

#for FBP
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
from skimage.transform import iradon
from skimage.transform import iradon_sart




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
parameters = [0.4, 1, 100, 12, 12.0] #r=4
#cameraman
#parameters = [1.2, 1, 381, 30, 8.0] #r=2

#parameters
n = 256 
k = parameters[1]
M = int(k*n)
N = n 

#angle set 'octant' types
BOW_TIE = -1
OCTANT = 0
FIRST_QUAD = 1
TOP_QUADS = 2

#OSEM params
s = parameters[3]
subsetsMode = 1
K = parameters[0]

iterations = parameters[2]
SNR_CT = 0.95
SNR_MRI = 40
floatType = np.float
twoQuads = True
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

# consts -----------------------------------------------------------------------
MRI_RECON = 1
CT_RECON = 0
SNR_CT = 0.95
SNR_MRI = 40
NUM_OCTANT_ANGLES = 25 + 1
ITERATIONS = 500
OCTANT_MRI = 2
OCTANT_CT = 4
LINE_MRI = "--"
LINE_CT = '-'

def get_compostie_sets(composites, class_type, recon_type=MRI_RECON): 
    """
    3 class types: 
        type 1: [q + pj, -q + pj] -> return equivalent angle in first oct and vertically mirrored
        type 2: [p + qj, -p + qj] -> return equivalent angle in second oct and vertically mirrored
        type 4: [q + pj,  p + qj, -p + qj, -q + pj] -> both
    """
    subsets = []
    for composite in composites: 
        p, q = farey.get_pq(composite)
        p, q = min(abs(p), abs(q)), max(abs(p), abs(q))
        if composite in [angle for subset in subsets for angle in subset]: 
            continue
        if class_type == 0: 
            angles = [farey.farey(p, q)]
        elif class_type == 1: 
            angles = [farey.farey(p, q), farey.farey(p, -1*q)]#type 1 - 1st and 4th octant
        elif class_type == 2: 
            angles = [farey.farey(q, p), farey.farey(q, -1*p)] #type 2 - 2nd and 3rd octant
        elif class_type == 4: 
            angles = [farey.farey(p, q), farey.farey(q, p), 
                            farey.farey(q, -1*p), farey.farey(p, -1*q)] #type 4 - all octants
        else: 
            return 
        
        if recon_type == MRI_RECON and class_type != 0: 
            angles = angles[0: len(angles) // 2]
        
        if angles not in subsets: 
            subsets.append(angles)
    
    

    return subsets


def dist_to_prime(primes, angle): 
    # angles, subsetAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles)    
    # primes, subsetPrimes, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles, prime_only=True)    
    # composites, subsetComposites = get_composites(subsetAngles)

    # distances = {}
    # for angle in composites: 

    min_dist = round(np.abs(np.angle(angle, deg=1) - np.angle(primes[0], deg=1)), 3)
    for prime in primes: 
        dist = np.abs(np.angle(angle, deg=1) - np.angle(prime, deg=1))
        dist = round(dist, 3)
        if dist < min_dist: 
            min_dist = dist
    return min_dist

    # if dist in distances.keys(): 
    #     distances[dist].append(angle)
    # else:
    #     distances[dist] = [angle]


def gcd_complex(n, m): 
    r_prev, r_n = m, n
    if abs(n) > abs(m): 
        r_prev, r_n = n, m

    while True: 
        gcd = r_n

        q_n = r_prev / r_n
        q_n = farey.farey(round(q_n.imag), round(q_n.real))
        r_n = r_prev - q_n * r_n
        r_n = farey.farey(r_n.imag, r_n.real)

        if abs(r_n) == 0:
            return gcd


def get_gaussian_prime_factors(angle): 
    """
    this works good enough

    remember that gaussian prime factorisation is unique apart from unit multiples
    
    https://stackoverflow.com/questions/2269810/whats-a-nice-method-to-factor-gaussian-integers#:~:text=In%20the%20Gaussian%20integers%2C%20if,2%20...%20pn.
    """
    p, q = farey.get_pq(angle)
    norm = p**2 + q**2
    primes = nt.factors(norm)

    factors = []
    for i, prime in enumerate(primes): 
        if prime == 2: 
            factor = farey.farey(1, 1)

        elif prime % 4 == 3: # only when q = 0, p !=0 or opposite
            factor = prime
            primes.pop(primes.index(prime, i+1)) #prime will be in factors twice, do not process twice. 

        else: # prime = 1 mod 4
            k = 0
            while k < 100: 
                if k ** 2 % prime == prime - 1:
                    break
                k += 1

            factor = gcd_complex(prime, farey.farey(1, k))
            # check if correct prime or require conjugate
            p, q = (angle / factor).imag, (angle / factor).real
            if not(p % 1 == 0 and q % 1 == 0):
                factor = factor.conjugate()
        factors.append(factor)
    
    return factors

#helpers -----------------------------------------------------------------------
def elNorm(l): 
    return lambda x: int(x.real**l+x.imag**l)
EUCLID_NORM = elNorm(2)
INF_NORM = lambda x: max(x.real, x.imag)

def remove_empty(subset_angles): 
    return [subset for subset in subset_angles if subset != []]


def add_noise(projs, snr=0.95):
    """Adds noise to the given projections. 

    Args:
        projs (list[list[ints]]): projections to ass noise to
        snr (int): Noise to add in decibels. Defaults to 0.95.
    """
    for m, proj in enumerate(projs):
        for t, bin in enumerate(proj):
            projs[m][t] = random.normalvariate(bin, 0.15*(1.0-snr)*bin)


def get_path(recon_type, recon_num, num_angles, iterations, noisy):
    path = "results_" + ("MRI" if recon_type else "CT") + "/"
    path += "recon_" + str(recon_num) + "/"
    path += "num_angles_" + str(num_angles) + "_"
    path += "its_" + str(iterations) + "_"
    path += "noise_" + str(noisy) + ".npz"
    return path 


# angle helpers ----------------------------------------------------------------
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


def get_composites(subsetAngles):
    subsetComposites = []
    composites = []
    for subset in subsetAngles:
        compositeSubset = []
        for angle in subset:
            if not (farey.is_gauss_prime(angle) or abs(angle) == 1):
                compositeSubset.append(angle)
                composites.append(angle)
        subsetComposites.append(compositeSubset)
    
    return composites, subsetComposites


def get_primes(subsetAngles):
    subsetPrimes = []
    primes = []
    for subset in subsetAngles:
        primeSubset = []
        for angle in subset:
            if farey.is_gauss_prime(angle) or abs(angle) == 1:
                primeSubset.append(angle)
                primes.append(angle)
        subsetPrimes.append(primeSubset)
    
    return primes, subsetPrimes


def closest_gaussian_prime(p_size, composite, num_to_store = 20): 
    """Identify a prime angle close to the given composite with a similar norm. 

    Args:
        p (int): prime size of image
        composite (complex): the composite angle to replace with a prime 
        num_to_store (int): number of 'close' primes to store before chosing 
        the close prime with the smallest norm  
    """

    def vector_angle(vector):
        p, q = farey.get_pq(vector)
        p, q = float(p), float(q)
        return p / q if q != 0 else 0

    fareyVectors = farey.Farey()
    fareyVectors.generatePrime(p_size-1, 1)
    vectors = fareyVectors.vectors

    p_neg, q_neg = farey.get_pq(composite)
    composite = farey.farey(abs(p_neg), abs(q_neg))
    angle_0 = vector_angle(composite)

    close_primes = []

    for oct_vector in vectors: 
        p, q = farey.get_pq(oct_vector)
        for vector in [oct_vector, farey.farey(q, p)]: #check angles in both octants 
            if len(close_primes) < num_to_store: #find closest 20 and choose that with the smallest norm 
                close_primes.append(vector)
                sorted(close_primes, key=lambda x: abs(vector_angle(x) - angle_0))
            else: 
                angle = vector_angle(vector)
                dist = abs(angle - angle_0)

                for i, prime in enumerate(close_primes): 
                    prime_angle = vector_angle(prime)

                    if dist < abs(prime_angle - angle_0): 
                        close_primes = close_primes[0:i] + [vector] + close_primes[i:-1]
                        break


    #sort to identify the smallest norm 
    close_primes = sorted(close_primes, key=EUCLID_NORM)  
    #adjust for correct quadrant 
    p, q = farey.get_pq(close_primes[0])
    if p_neg < 0: 
        p = -1 * p
    if q_neg < 0: 
        q = -1 * q

    return farey.farey(p, q)


def plot_angles(angles, colour='skyblue', line='-', linewidth=1, label="angles", ax=None): 
    """
    plot from origin to vector/angle, like the sunbeam
    
    angles (list[complex]): angles to plot
    line (str): colour and line type to plot 
    """
    if ax: 
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 1])
    else:
        ax = plt
        ax.xlim((-1, 1))
        ax.ylim((0, 1))

    for angle in angles: 
        imag, real = farey.get_pq(angle)
        ax.plot([0, real], [0, imag], line, c=colour, linewidth=linewidth, label=label)

# FBP --------------------------------------------------------------------------
def fbp(p, num_angles, noisy=False):
    image = imread(data_dir + "/phantom.png", as_grey=True)
    image = rescale(image, scale = float(p) / 400, mode='constant')
    theta = np.linspace(0., 180., num_angles, endpoint=True)
    sinogram = radon(image, theta=theta, circle=True)

    if noisy: 
        add_noise(sinogram, snr=SNR_CT)

    data = {}

    reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)
    rmse_fbp = np.sqrt(imageio.immse(image, reconstruction_fbp))
    psnr_fbp = imageio.impsnr(image, reconstruction_fbp)
    ssim_fbp = imageio.imssim(image.astype(float), reconstruction_fbp.astype(float))
    data["fbp"] = {"rmse":rmse_fbp, "psnr":psnr_fbp, "ssim":ssim_fbp}

    reconstruction_sart = iradon_sart(sinogram, theta=theta)
    rmse_sart_1 = np.sqrt(imageio.immse(image, reconstruction_sart))
    psnr_sart_1 = imageio.impsnr(image, reconstruction_sart)
    ssim_sart_1 = imageio.imssim(image.astype(float), reconstruction_sart.astype(float))
    data["sart1"] = {"rmse":rmse_sart_1, "psnr":psnr_sart_1, "ssim":ssim_sart_1}


    reconstruction_sart2 = iradon_sart(sinogram, theta=theta,
                                    image=reconstruction_sart)
    rmse_sart_2 = np.sqrt(imageio.immse(image, reconstruction_sart2))
    psnr_sart_2 = imageio.impsnr(image, reconstruction_sart2)
    ssim_sart_2 = imageio.imssim(image.astype(float), reconstruction_sart2.astype(float))
    data["sart2"] = {"rmse":rmse_sart_2, "psnr":psnr_sart_2, "ssim":ssim_sart_2}

    return reconstruction_sart2, rmse_sart_2, psnr_sart_2, ssim_sart_2


# Base for CT and MRI reconstructions ------------------------------------------
def angleSubSets_Symmetric(s, mode, P, Q, octant=1, binLengths=False, K = 1, prime_only = False, max_angles = 10, norm=EUCLID_NORM):
    '''
    Generate the minimal L1 angle set for the MT for s subsets.
    Parameter K controls the redundancy, K = 1 is minimal.
    If octant is number of octancts to use. 
    Function can also return bin lengths for each bin.
    '''
    angles = []
    subsetAngles = []
    for i in range(s):
        subsetAngles.append([])
    fareyVectors = farey.Farey()
    maxPQ = max(P,Q)

    fareyVectors.compactOff()
    if prime_only:
        fareyVectors.generatePrime(maxPQ-1, 1)
        print("primes")
    else: 
        fareyVectors.generate(maxPQ-1, 1)
        
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=norm) 

    index = 0
    num_angles = 0
    subsetIndex = 0
    binLengthList = []

    if max_angles < 0: 
        print("Katz!")
        angle_cond = lambda angles, _: not mojette.isKatzCriterion(P, Q, angles, K)
    else: 
        angle_cond = lambda _, num_angles: num_angles < max_angles - 1

    while index < len(sortedVectors) and angle_cond(angles, num_angles): # check Katz
    #     index += 1
        angle = sortedVectors[index]
        angles.append(angle)
        subsetAngles[subsetIndex].append(angle)
        index += 1
        num_angles += 1
        p, q = farey.get_pq(angle) # p = imag, q = real
        # print(p, q)

        if octant > 1 and p != q:
            nextOctantAngle = farey.farey(q, p) #swap to mirror from diagonal
            angles.append(nextOctantAngle)
            subsetAngles[subsetIndex].append(nextOctantAngle)
            num_angles += 1
            binLengthList.append(mojette.projectionLength(nextOctantAngle,P,Q))
            if mode == 1:
                subsetIndex += 1
                subsetIndex %= s

        if p == 0 or q == 0: 
            continue #the only case should be 0 + 1j and 1 + 0j

        if octant > 2:
            nextOctantAngle = farey.farey(q, -p) #third octant
            angles.append(nextOctantAngle)
            subsetAngles[subsetIndex].append(nextOctantAngle)
            num_angles += 1
            binLengthList.append(mojette.projectionLength(nextOctantAngle,P,Q))
            if mode == 1:
                subsetIndex += 1
                subsetIndex %= s

            if p != q: #dont replicate
                nextOctantAngle = farey.farey(p, -q) #third octant
                angles.append(nextOctantAngle)
                subsetAngles[subsetIndex].append(nextOctantAngle)
                num_angles += 1
                binLengthList.append(mojette.projectionLength(nextOctantAngle,P,Q))
                if mode == 1:
                    subsetIndex += 1
                    subsetIndex %= s
        if mode == 0:
            subsetIndex += 1
            subsetIndex %= s

    if binLengths:
        return angles, subsetAngles, binLengthList
    
    return angles, subsetAngles


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
    fdtype = np.float
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
                # print("Smooth TV")
                f = denoise_tv_chambolle(f, 0.02, multichannel=False)
            elif smoothReconMode == 2:
                h = parameters[4]
                if i > smoothMaxIteration:
                    h /= 2.0
                if i > smoothMaxIteration2:
                    h /= 4.0
                # print("Smooth NL h:",h)
                fReal = denoise_nl_means(np.real(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                # fImag = denoise_nl_means(np.imag(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                f = fReal #+1j*fImag
            elif smoothReconMode == 3:
                # print("Smooth Median")
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
            print("i:", i, "RMSE:", math.sqrt(mse), "PSNR:", psnr, "SSIM:", ssim, end="\r")
            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
    print()    
    return f, mses, psnrs, ssims


def recon_CT(p, angles, subsetAngles, iterations, noisy=False): 
    lena, mask = imageio.phantom(N, p, True, np.uint32, True)

    #convert angles to gradients for OSEM
    subsetsMValues = []
    for subset in subsetAngles:
        mValues = []
        for angle in subset:
            m, inv = farey.toFinite(angle, p)
            mValues.append(m)            
        subsetsMValues.append(mValues)
    

    mt_lena = mojette.transform(lena, angles)
    
    #add noise 
    if noisy:
        add_noise(mt_lena, SNR_CT) 

    rt_lena = mojette.toDRT(mt_lena, angles, p, p, p) 
    
    recon, mses, psnrs, ssims = osem_expand(iterations, p, rt_lena, subsetsMValues, finite.frt, finite.ifrt, lena, mask)
    recon = np.abs(recon)

    return recon, np.sqrt(mses), psnrs, ssims


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
    fdtype = np.complex
    f = np.ones((p,p), fdtype)
    
    mses = []
    psnrs = []
    ssims = []
    for i in range(0, iterations):
        # print("Iteration:", i)
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
                # print("Smooth TV")
                f = denoise_tv_chambolle(f, 0.02, multichannel=False)
            elif smoothReconMode == 2:
                h = parameters[4]
                if i > smoothMaxIteration:
                    h /= 2.0
                if i > smoothMaxIteration2:
                    h /= 4.0
                # print("Smooth NL h:",h)
                fReal = denoise_nl_means(np.real(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                fImag = denoise_nl_means(np.imag(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                f = fReal +1j*fImag
            elif smoothReconMode == 3:
                # print("Smooth Median")
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
            print("i:", i, "RMSE:", math.sqrt(mse), "PSNR:", psnr, "SSIM:", ssim, end="\r")
            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
    print()   
    return f, mses, psnrs, ssims


def recon_MRI(p, angles, subsetAngles, iterations, noisy=False): 
    image, mask = imageio.phantom(N, p, True, np.uint32, True)

    #k-space
    fftImage = fftpack.fft2(image) #the '2' is important
    fftImageShifted = fftpack.fftshift(fftImage)

    #power spectrum
    powSpectImage = np.abs(fftImageShifted)

    #add noise to kSpace
    noise = finite.noise(fftImageShifted, SNR_MRI)
    if noisy:
        fftImageShifted += noise

    #Recover full image with noise
    reconImage = fftpack.ifft2(fftImageShifted) #the '2' is important
    reconImage = np.abs(reconImage)
    reconNoise = image - reconImage

    mse = imageio.immse(image, np.abs(reconImage))
    ssim = imageio.imssim(image.astype(float), np.abs(reconImage).astype(float))
    psnr = imageio.impsnr(image, np.abs(reconImage))

    #compute lines
    centered = True
    subsetsLines = []
    subsetsMValues = []
    mu = 0
    for subset in subsetAngles:
        lines = []
        mValues = []
        for angle in subset:
            m, inv = farey.toFinite(angle, p)
            u, v = radon.getSliceCoordinates2(m, powSpectImage, centered, p)
            lines.append((u,v))
            mValues.append(m)
            #second quadrant
            if twoQuads:
                if m != 0 and m != p: #dont repeat these
                    m = p-m
                    u, v = radon.getSliceCoordinates2(m, powSpectImage, centered, p)
                    lines.append((u,v))
                    mValues.append(m)
        subsetsLines.append(lines)
        subsetsMValues.append(mValues)
        mu += len(lines)
    #samples used
    sampleNumber = (p-1)*mu
    #-------------
    # Measure finite slice

    drtSpace = np.zeros((p+1, p), np.complex)
    for lines, mValues in zip(subsetsLines, subsetsMValues):
        for i, line in enumerate(lines):
            u, v = line
            sliceReal = ndimage.map_coordinates(np.real(fftImageShifted), [u,v])
            sliceImag = ndimage.map_coordinates(np.imag(fftImageShifted), [u,v])
            slice = sliceReal+1j*sliceImag
            finiteProjection = fftpack.ifft(slice) # recover projection using slice theorem
            drtSpace[mValues[i],:] = finiteProjection

    recon, mses, psnrs, ssims = osem_expand_complex(iterations, p, drtSpace, subsetsMValues, finite.frt_complex, finite.ifrt_complex, image, mask)

    return recon, mses, psnrs, ssims



# base angle set reconstructions -----------------------------------------------
def regular_recon(p, num_angles_octant, iterations, recon_type=MRI_RECON, colour="hotpink", line="-", noisy=False):
    """Completes one MRI or CT reconstruction. Plots error info. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon (int, optional): Specificy MRI recon (=1) or CT recon (=0). 
        Defaults to 0.
        colour (str, optional): Colour of plot. Defaults to "hotpink".
    """
    #num angles given to each reconstruction. note: number CT uses is the number 
    #of projections for both reconstructions. 
    num_angles_mri = num_angles_octant * OCTANT_MRI - 1 #must be odd to account for (1, 1) having no mirror pair  
    num_angles_ct = 2 * (num_angles_mri - 2) #each angle maps to itself and its vertical mirror EXCEPT (0, 1) and (1, 0) 

    if recon_type: #MRI RECON
        num_angles = num_angles_mri if num_angles_octant > 0 else -1
        octant = OCTANT_MRI
        recon = recon_MRI
        path_head = "results_MRI/"
        title = "MRI reconstruction"
    else: #CT RECON
        num_angles = num_angles_ct if num_angles_octant > 0 else -1
        octant = OCTANT_CT
        recon = recon_CT
        path_head = "results_CT/"
        title = "CT reconstruction"

    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=octant,K=K, max_angles=num_angles)  

    recon_im, rmses, psnrs, ssims = recon(p, angles, remove_empty(subsetAngles), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour=colour, line=line, label="regular recon, " + str(num_angles_ct) + " projections")
    # plt.suptitle(title)

    return angles, recon_im, rmses, psnrs, ssims


def prime_recon(p, num_angles_octant, iterations, recon_type=MRI_RECON, colour="skyblue", line="-", noisy=False): 
    """Completes one MRI or CT reconstruction with only the prime angle set. 
    Plots error info. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon (int, optional): Specificy MRI recon (=1) or CT recon (=0). 
        Defaults to 0
    """
    #num angles given to each reconstruction. note: number CT uses is the number 
    #of projections for both reconstructions. 
    num_angles_mri = num_angles_octant * OCTANT_MRI - 1 #must be odd to account for (1, 1) having no mirror pair  
    num_angles_ct = 2 * (num_angles_mri - 2) #each angle maps to itself and its vertical mirror EXCEPT (0, 1) and (1, 0) 

    if recon_type: #MRI RECON
        num_angles = num_angles_mri
        octant = OCTANT_MRI
        recon = recon_MRI
        path_head = "results_MRI/"
        title = "MRI prime reconstruction"
    else: #CT RECON
        num_angles = num_angles_ct
        octant = OCTANT_CT
        recon = recon_CT
        path_head = "results_CT/"
        title = "CT prime reconstruction"

    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=octant,K=K, max_angles=num_angles) 
    primes, primes_subset = get_primes(subset_angles)
    recon_im, rmses, psnrs, ssims = recon(p, primes, remove_empty(primes_subset), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour=colour, line=line, label="prime recon, " + str(num_angles_ct) + " projections")

    return angles, recon_im, rmses, psnrs, ssims


def composite_recon(p, num_angles_octant, iterations, recon_type=MRI_RECON, colour="mediumpurple", line="-", noisy=False): 
    """Completes one MRI or CT reconstruction with only the composite angle set. 
    Plots error info. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon (int, optional): Specificy MRI recon (=1) or CT recon (=0). 
        Defaults to 0
    """
    num_angles_mri = num_angles_octant * OCTANT_MRI - 1 #must be odd to account for (1, 1) having no mirror pair  
    num_angles_ct = 2 * (num_angles_mri - 2) #each angle maps to itself and its vertical mirror EXCEPT (0, 1) and (1, 0) 

    if recon_type: #MRI RECON
        num_angles = num_angles_mri
        octant = OCTANT_MRI
        recon = recon_MRI
        path_head = "results_MRI/"
        title = "MRI composite reconstruction"
    else: #CT RECON
        num_angles = num_angles_ct
        octant = OCTANT_CT
        recon = recon_CT
        path_head = "results_CT/"
        title = "CT composite reconstruction"

    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=octant,K=K, max_angles=num_angles) 
    comps, comps_subset = get_composites(subset_angles)
    recon_im, rmses, psnrs, ssims = recon(p, comps, remove_empty(comps_subset), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour=colour, line=line, label="composite recon, " + str(num_angles_ct) + " projections")

    return angles, recon_im, rmses, psnrs, ssims


def comp_recplacement_recon(p, num_angles_octant, iterations, num_to_store=20, recon_type=MRI_RECON, colour="limegreen", line="-", noisy=False):
    """Completes one MRI or CT reconstruction with the all composite angles in 
    the angle set replaced with the closest prime angle. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon_type (int, optional): Specificy MRI recon (=1) or CT recon (=0). Defaults to MRI_RECON.
        colour (str, optional): colour to plot. Defaults to "limegreen".
        line (str, optional): line style to plot. Defaults to "-".
        noisy (bool, optional): reconstruct with or without noise. Defaults to False.
    """

    num_angles_mri = num_angles_octant * OCTANT_MRI - 1 #must be odd to account for (1, 1) having no mirror pair  
    num_angles_ct = 2 * (num_angles_mri - 2) #each angle maps to itself and its vertical mirror EXCEPT (0, 1) and (1, 0) 

    if recon_type: #MRI RECON
        num_angles = num_angles_mri
        octant = OCTANT_MRI
        recon = recon_MRI
    else: #CT RECON
        num_angles = num_angles_ct
        octant = OCTANT_CT
        recon = recon_CT

    
    #prime replacement recon
    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=octant,K=K, max_angles=num_angles) 

    comp_replacements = []
    for i, subset in enumerate(subset_angles): 
        for j, angle in enumerate(subset): 
            if not (farey.is_gauss_prime(angle) or abs(angle) == 1): 
                prime = closest_gaussian_prime(p, angle, num_to_store)
                subset_angles[i][j] = prime
                comp_replacements.append(prime)

    angles = [angle for subset in subset_angles for angle in subset]
    
    recon_im, rmses, psnrs, ssims = recon(p, angles, remove_empty(subset_angles), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour=colour, line="--", label="prime replacement, " + str(num_angles_ct) + " projections")
    
    return angles, recon_im, rmses, psnrs, ssims


#fractal functions -------------------------------------------------------------
def calcFiniteLines(angles): 
    powerSpect = np.zeros((p,p))
    centreed = True
    lines = []
    mValues = []

    for angle in angles:
        m, inv = farey.toFinite(angle, p)
        u, v = radon.getSliceCoordinates2(m, powerSpect, centreed, p)
        lines.append((u,v))
        mValues.append(m)
        #second quadrant
        if twoQuads:
            if m != 0 and m != p: #dont repeat these
                m = p-m
                u, v = radon.getSliceCoordinates2(m, powerSpect, centreed, p)
                lines.append((u,v))
                mValues.append(m)
    
    return (lines, mValues)


def createFractal(lines, p): 
    powerSpect = np.zeros((p,p))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    plt.gray()
    plt.tight_layout()

    maxLines = len(lines)
    #maxLines = 12
    ax[0].imshow(powerSpect)
    ax[1].imshow(powerSpect)
    #color=iter(cm.rainbow(np.linspace(0,1,len(lines))))
    color=iter(plt.cm.jet(np.linspace(0,1,maxLines+1)))
    fareyImage = np.zeros_like(powerSpect)
    for i, line in enumerate(lines):
        u, v = line
        c=next(color)
        ax[0].plot(u, v, '.', c=c)
        ax[1].plot(u, v, '.r',markersize=1)
        fareyImage[u,v] = 255
        if i == maxLines:
            break
    ax[0].set_title('Sampling (colour per line) for prime size:'+str(p))
    ax[1].set_title('Sampling (same colour per line) for prime size:'+str(p))


def plotFractal(angles, recon_type, title="fractal", num_to_store=0): 
    lines, mValues = calcFiniteLines(angles)
    createFractal(lines, p)
    path = "result_" + ("MRI" if recon_type else "CT") + '/'
    path += "recon_3/num_to_store_" + str(num_to_store) + ".png"
    plt.savefig(path)


# reconstructions --------------------------------------------------------------
def recon_neg_2(p, iterations, num_angles): 
    """Reconstructs images via ChaoS and FBP for a comparrison of the two methods. 
    Reconstructs methods with and without noise. Saves reconstruction and errors. 
    To reconstruct, see plot_neg_2. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
    """
    data = {}
    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=num_angles) 

    recon_im, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, False)
    data["no noise"] = {"recon": recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims}
    # plot_recon(rmses, psnrs, ssims, colour="hotpink", line="-", label="not noisy recon, " + str(num_angles) + " projections")

    recon_im_noisy, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, True)
    # plot_recon(rmses, psnrs, ssims, colour="skyblue", line="--", label="noisy recon, " + str(num_angles) + " projections")
    data["noise"] = {"recon": recon_im_noisy, "rmse": rmses, "psnr": psnrs, "ssim": ssims}


    recon_im_fbp, rmse, psnr, ssim = fbp(p, num_angles)
    rmses = rmse * np.ones_like(rmses)
    psnrs = psnr * np.ones_like(psnrs)
    ssims = ssim * np.ones_like(ssims)
    data["FBP no noise"] = {"recon": recon_im_fbp, "rmse": rmses, "psnr": psnrs, "ssim": ssims}

    recon_im_fbp_noisy, rmse, psnr, ssim = fbp(p, num_angles, noisy=True)
    rmses = rmse * np.ones_like(rmses)
    psnrs = psnr * np.ones_like(psnrs)
    ssims = ssim * np.ones_like(ssims)
    data["FBP noise"] = {"recon": recon_im_fbp_noisy, "rmse": rmses, "psnr": psnrs, "ssim": ssims}


    np.savez(file="results_CT/recon_neg_2/FBP_ChaoS_num_angles_" + str(num_angles) + ".npz", data=data)


def recon_neg_1(p, iterations):
    """ 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon (int, optional): Specificy MRI recon (=1) or CT recon (=0). 
        Defaults to 0.
        colour (str, optional): Colour of plot. Defaults to "hotpink".
    """
    num_angles = 257
    octant = OCTANT_CT
    recon = recon_CT
    path_head = "results_CT/"
    title = "CT reconstruction"

    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=octant,K=K, max_angles=num_angles)  

    recon_im, rmses, psnrs, ssims = recon(p, angles, remove_empty(subsetAngles), iterations, noisy=True)
    plot_recon(rmses, psnrs, ssims, colour="hotpink", line="-", label="ChaoS not noisy recon, " + str(num_angles) + " projections")

    recon_im, rmses, psnrs, ssims = recon(p, angles, remove_empty(subsetAngles), iterations, noisy=False)
    plot_recon(rmses, psnrs, ssims, colour="skyblue", line="-", label="ChaoS noisy recon, " + str(num_angles) + " projections")


def recon_0(p, num_angles_octant, iterations, noisy=False):
    """Completes reconstruction of MRI and CT for same parameters. Plots error info. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon (int, optional): Specificy MRI recon (=1) or CT recon (=0). 
        Defaults to 0.
        colour (str, optional): Colour of plot. Defaults to "hotpink".
    """
    data = {}    
    angles, recon, rmses, psnrs, ssims = regular_recon(p, num_angles_octant, iterations, recon_type=MRI_RECON, colour="skyblue", line="--", noisy=noisy)
    data["regular"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
    path = get_path(MRI_RECON, 0, num_angles_octant, ITERATIONS, noisy)
    print("num MRI angles:", len(angles))
    np.savez(file=path, data=data)

    data = {}
    angles, recon, rmses, psnrs, ssims = regular_recon(p, num_angles_octant, iterations, recon_type=CT_RECON, colour="hotpink", line="-", noisy=noisy)
    data["regular"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
    path = get_path(CT_RECON, 0, num_angles_octant, ITERATIONS, noisy)
    print("num CT angles:", len(angles))
    # np.savez(file=path, data=data)


def recon_1(p, num_angles_octant, iterations, recon_type=MRI_RECON, noisy=False): 
    """Reconstructions of regular, prime, and composite angle sets for CT or MRI. 
    Plots and saves. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon (int, optional): Specificy MRI recon (=1) or CT recon (=0). 
        Defaults to 0
    """
    data = {}
    
    angles, recon, rmses, psnrs, ssims = regular_recon(p, num_angles_octant, iterations, recon_type, noisy=noisy)
    data["regular"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    angles, recon, rmses, psnrs, ssims = prime_recon(p, num_angles_octant, iterations, recon_type, noisy=noisy)
    data["prime"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    angles, recon, rmses, psnrs, ssims = composite_recon(p, num_angles_octant, iterations, recon_type, noisy=noisy)
    data["composite"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    path = get_path(recon_type, 1, num_angles_octant, iterations, noisy)
    np.savez(file=path, data=data)


def recon_2(p, num_angles_octant, iterations, recon_type=MRI_RECON, noisy=False): 

    #angles for top two quadrants
    num_angles_MRI = 2 * num_angles_octant - 1  
    num_angles_CT = 2 * (num_angles_MRI - 2)

    if recon_type: #MRI RECON
        num_angles = num_angles_MRI
        octant = OCTANT_MRI
        recon = recon_MRI
        title = "MRI reconstruction"
    else: #CT RECON
        num_angles = num_angles_CT
        octant = OCTANT_CT
        recon = recon_CT
        title = "CT reconstruction"

    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=octant,K=K, max_angles=num_angles) 
    composites, subset_composites = get_composites(subset_angles)
    
    #set up dictionary pre recon 
    data = {}
    for [comp] in get_compostie_sets(composites, 0): 
        data[comp] = {}

    #reconstruct CT and MRI for the three different equivalent classes 
    for equiv_class_type in [1, 2, 4]: 
        equiv_comp_classes = get_compostie_sets(composites, equiv_class_type, recon_type)

        plt.figure(figsize=(16, 8))
        plt.suptitle(title)
        colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(equiv_comp_classes)+3)))

        for i, comp_equiv_class in enumerate(equiv_comp_classes):
            
            #reset angle set to prime angles
            new_angles, new_subset_angles = get_primes(subset_angles) 
            
            #add composite angles to prime angle set 
            for i, angle in enumerate(comp_equiv_class):
                #add at same index as in regular subset
                idx = get_subset_index(angle, subset_composites)
                new_subset_angles[idx].append(angle)
                new_angles.append(angle)

            #reconstruct with new compositie + prime subset 
            recon_im, rmses, psnrs, ssims = recon(p, new_angles, remove_empty(new_subset_angles), iterations, noisy)
            plot_recon(rmses, psnrs, ssims, label=str(comp_equiv_class), colour=next(colour))

            #use first octant as key 
            a, b = farey.get_pq(comp_equiv_class[0])
            comp = farey.farey(min(abs(a), abs(b)), max(abs(a), abs(b)))
            data[comp][equiv_class_type] = {"angles": new_angles, "recon":recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
    
    path = get_path(recon_type, 2, num_angles, iterations, noisy)
    np.savez(file=path, data=data)

    # plt.savefig("result_MRI_CT/recon_2/angles_" + str(num_angles_octant) + "_its_" + str(iterations) + "_rev2.png")


def recon_3(p, num_angles_octant, iterations, recon_type=MRI_RECON, noisy=False):
    """CT reconstruction for regular angle set, and angle set with all comopsite 
    angles replaced with prime angles. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon_type (int, optional): Selects MRI or CT reconstruction. Defaults to MRI_RECON.
        noisy (bool, optional): Selects to add noise to reconstruction. Defaults to False.
    """
    colour = iter(plt.cm.gist_rainbow(np.linspace(0,1, 5 + 1)))
    data = {}

    angles, recon, rmses, psnrs, ssims = regular_recon(p, num_angles_octant, iterations, recon_type, noisy=noisy)
    data["regular"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    data["prime_replacement"] = {}
    for num_to_store in [5, 10, 20, 30]: 
        angles, recon, rmses, psnrs, ssims = comp_recplacement_recon(p, num_angles_octant, iterations, num_to_store=num_to_store, recon_type=recon_type, colour="skyblue", noisy=noisy)
        data["prime_replacement"][num_to_store] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    path = get_path(recon_type, 3, num_angles_octant, iterations, noisy)
    np.savez(file=path, data=data)
    

# additional recons ------------------------------------------------------------
def recon_1b(p, iterations, recon_type=MRI_RECON, noisy=False): 
    """Runs reconstruction for an increasing number of angles. 

    Args:
        p (int): prime size of image
        iterations (int): number of OSEM iterations
        recon_type (int, optional): Selects MRI or CT reconstruction. Defaults to MRI_RECON.
        noisy (bool, optional): Selects to add noise to reconstruction. Defaults to False.
    """
    data = {"regular":{}, "prime":{}}
    octant_angles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]

    plt.figure(figsize=(16, 8))
    colour = iter(plt.cm.gist_rainbow(np.linspace(0,1, 2 * len(octant_angles) + 1)))

    for num_angles in octant_angles: 
        angles, recon_im, rmses, psnrs, ssims = regular_recon(p, num_angles, iterations, recon_type, colour=next(colour), noisy=noisy)
        data["regular"][num_angles] = {"angles": angles, "recon": recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

        angles, recon_im, rmses, psnrs, ssims = prime_recon(p, num_angles, iterations, recon_type, colour=next(colour), noisy=noisy)
        data["prime"][num_angles] = {"angles": angles, "recon": recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    path = "results_CT/recon_1/recon_1b/many_angle_its_" + str(iterations) 
    np.savez(file=path + ".npz", data=data)
    plt.savefig(path + ".png")

    
# plotters ---------------------------------------------------------------------
def plot_recon(rmseValues, psnrValues, ssimValues, colour = "b", line = '-', label="label"):
    incX = np.arange(0, len(rmseValues))*plotIncrement

    plt.subplot(1, 3, 1)
    plt.plot(incX, rmseValues, c=colour, ls=line, label=label)
    plt.title('Error Convergence of the Finite OSEM')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')

    plt.subplot(1, 3, 2)
    plt.plot(incX, ssimValues, c=colour, ls=line, label=label)
    plt.ylim(0, 1.0)
    plt.title('Simarlity Convergence of the Finite OSSEM')
    plt.xlabel('Iterations')
    plt.ylabel('SSIM')

    plt.subplot(1, 3, 3)
    plt.plot(incX, psnrValues, c=colour, ls=line, label=label)
    plt.ylim(0, 45.0)
    plt.title('PSNR Convergence of the Finite OSSEM')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.legend()


def plot_neg_2(path): 
    data = np.load(path)["data"].item()

    #plot errors
    plt.figure(figsize=(16, 8))
    plot_recon(data["no noise"]["rmse"], data["no noise"]["psnr"], data["no noise"]["ssim"], label="ChoaS no noise", colour="hotpink")
    plot_recon(data["noise"]["rmse"], data["noise"]["psnr"], data["noise"]["ssim"], label="ChoaS noise", colour="hotpink", line='--')
    # plot_recon(data["FBP no noise"]["rmse"], data["FBP no noise"]["psnr"], data["FBP no noise"]["ssim"], label="FBP no noise", colour="mediumpurple")
    plot_recon(data["FBP noise"]["rmse"], data["FBP noise"]["psnr"], data["FBP noise"]["ssim"], label="FBP noise", colour="mediumpurple", line='--')


    #plot recons
    recon_im = data["no noise"]["recon"]
    recon_im_noisy = data["noise"]["recon"]
    recon_im_fbp = data["FBP no noise"]["recon"]
    recon_im_fbp_noisy = data["FBP noise"]["recon"]

    lena, mask = imageio.phantom(N, p, True, np.uint32, True)
    lena_fbp = imread(data_dir + "/phantom.png", as_grey=True)
    lena_fbp = rescale(lena_fbp, scale = float(p) / 400, mode='constant')

    fig, axs = plt.subplots(2, 4, figsize=(18, 12), sharex=True, sharey=True)
    axs = axs.flat
    axs[0].imshow(recon_im, cmap="gray")
    axs[1].imshow(abs(recon_im - lena), cmap="gray")
    axs[2].imshow(recon_im, cmap="gray")
    axs[3].imshow(abs(recon_im_noisy - lena), cmap="gray")
    axs[4].imshow(recon_im_fbp, cmap="gray")
    axs[5].imshow(abs(recon_im_fbp - lena_fbp), cmap="gray")
    axs[6].imshow(recon_im_fbp_noisy, cmap="gray")
    axs[7].imshow(abs(recon_im_fbp_noisy - lena_fbp), cmap="gray")


def plot_recon_1b(): 
    """Plots reg and prime recons for increasing number of angles for octant. 
    """
    path = "results_CT/recon_1/recon_1b/many_angle_its_500.npz"
    data = np.load(path)["data"].item()
    reg_recon_info = data["regular"]
    prime_recon_info = data["prime"]

    octant_angles = reg_recon_info.keys()
    octant_angles = [10, 20, 30, 40, 100]

    colour = iter(plt.cm.gist_rainbow(np.linspace(0,1, len(octant_angles) + 1)))

    for num_angles in octant_angles: 
        c = next(colour)

        reg_recon = reg_recon_info[num_angles]
        plot_recon(reg_recon["rmse"], reg_recon["psnr"], reg_recon["ssim"], 
                   colour = c, label="reg, " + str(num_angles) + " projs")

        prime_recon = prime_recon_info[num_angles]
        plot_recon(prime_recon["rmse"], prime_recon["psnr"], prime_recon["ssim"], 
                   colour = c, line="--", label="prime, " + str(num_angles) + " projs")
    

def plot_recon_2(path, plot_angle=True, plot_type=True):
    data = np.load(path)["data_MRI"].item()
    equiv_class_types = [1, 2, 4]
    if plot_angle:
        for angle in data.keys(): 
            plt.figure(figsize=(16, 8))
            colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(equiv_class_types+1))))
            for equiv_class_type in equiv_class_types: 
                error_info = data[angle][equiv_class_type]
                plot_recon(error_info["rmse"], error_info["psnr"], error_info["ssim"], label = str(angle) + " type " + str(equiv_class_type), colour = next(colour))

    if plot_type: 
        for equiv_class_type in equiv_class_types: 
            plt.figure(figsize=(16, 8))
            colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(data.keys())+1)))
            for angle in data.keys(): 
                error_info = data[angle][equiv_class_type]
                plot_recon(error_info["rmse"], error_info["psnr"], error_info["ssim"], label = str(angle) + " type " + str(equiv_class_type), colour = next(colour))

    plt.show()

 
# questions and new ------------------------------------------------------------

def ct_katz(p, iterations, noisy=False):
    """Completes reconstruction of MRI and CT for same parameters. Plots error info. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon (int, optional): Specificy MRI recon (=1) or CT recon (=0). 
        Defaults to 0.
        colour (str, optional): Colour of plot. Defaults to "hotpink".
    """
    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=62)  
    recon_im, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour="hotpink", label="regular recon, " + str(len(angles)) + " projections")

    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=62, prime_only=True)  
    recon_im, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour="skyblue", label="prime recon, " + str(len(angles)) + " projections")
    # plt.suptitle(title)


#Shes a runner shes a track star -----------------------------------------------
if __name__ == "__main__": 
    p = nt.nearestPrime(N)

    # for num_angle in [25, 50, 75, 100, 125]: 
    #     plt.figure(figsize=(16, 8))
    #     recon_neg_2(p, 300, num_angle)

    for num_angle in [25, 50, 75, 100, 125]: 
        path = "results_CT/recon_neg_2/FBP_ChaoS_num_angles_{}.npz".format(num_angle)
        plot_neg_2(path)

    # plt.figure(figsize=(16, 8))
    # recon_neg_1(p, ITERATIONS)

    # for num_angles in [25, 50, 75, 100]: 
    #     plt.figure(figsize=(16, 8))
    #     recon_1(p, num_angles, ITERATIONS, CT_RECON, False)

    #     plt.figure(figsize=(16, 8))
    #     recon_1(p, num_angles, ITERATIONS, CT_RECON, True)

    # plt.figure(figsize=(16, 8))
    # recon_2(p, NUM_OCTANT_ANGLES, ITERATIONS, CT_RECON, noisy=False)

    # plt.figure(figsize=(16, 8))
    # recon_2(p, NUM_OCTANT_ANGLES, ITERATIONS, CT_RECON, noisy=True)

    plt.show()




    # ct_katz(p, ITERATIONS, False)

    # path = "results_CT/recon_2/num_angles_78_its_500_noise_False.npz"
    # data = np.load(path)["data"].item()
    # print(data.keys())


    # recon_1b(p, ITERATIONS, CT_RECON, noisy=False)
    # plt.figure(figsize=(16, 8))
    # recon_2(p, NUM_OCTANT_ANGLES, ITERATIONS, CT_RECON, noisy=False)
    # plt.figure(figsize=(16, 8))
    # recon_3(p, NUM_OCTANT_ANGLES, ITERATIONS, CT_RECON, noisy=False)
    # plt.show()
    
# %%
