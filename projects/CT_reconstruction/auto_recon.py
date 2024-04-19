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
s = 12
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

INF_NORM = lambda x: max(x.real, x.imag)
def elNorm(l): 
    return lambda x: int(x.real**l+x.imag**l)
EUCLID_NORM = elNorm(2)

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


def get_compostie_sets(comps, type): 
    """
    3 types of composite angle sets: 
        type 1: [q + pj, -q + pj] -> return equivalent angle in first oct and vertically mirrored
        type 2: [p + qj, -p + qj] -> return equivalent angle in second oct and vertically mirrored
        type 4: [q + pj,  p + qj, -p + qj, -q + pj] -> both
    """
    subsets = []
    for comp in comps: 
        p, q = farey.get_pq(comp)
        p, q = min(abs(p), abs(q)), max(abs(p), abs(q))
        if comp in [angle for subset in subsets for angle in subset]: 
            continue
        if type == 0: 
            angles = [farey.farey(p, q)]
        elif type == 1: 
            angles = [farey.farey(p, q), farey.farey(p, -1*q)]#type 1 - mirror vertical
        elif type == 2: 
            angles = [farey.farey(q, p), farey.farey(q, -1*p)] #type 2 - + 90deg
        elif type == 4: 
            angles = [farey.farey(p, q), farey.farey(q, p), 
                            farey.farey(q, -1*p), farey.farey(p, -1*q)] #type 4 - all quads
        else: 
            return 
        
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

    while index < len(sortedVectors) and num_angles < max_angles - 1: # check Katz
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
    print("Noise:", noisy)
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


#helpers -----------------------------------------------------------------------
def remove_empty(subset_angles): 
    return [subset for subset in subset_angles if subset != []]


# angle helpers ----------------------------------------------------------------
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


def closest_gaussian_prime(p_size, composite): 
    """Identify a prime angle close to the given composite with a similar norm. 

    Args:
        p (int): prime size of image
        composite (complex): the composite angle to replace with a prime 
    """

    def vector_angle(vector):
        p, q = farey.get_pq(vector)
        p, q = float(p), float(q)
        return p / q if q != 0 else 0

    fareyVectors = farey.Farey()
    fareyVectors.generatePrime(p_size-1, 1)
    vectors = fareyVectors.vectors

    close_primes = []
    angle_0 = vector_angle(composite)

    for vector in vectors: 
        if len(close_primes) < 20: #find closest 20 and choose that with the smallest norm 
            close_primes.append(vector)
            sorted(close_primes, key=lambda x: abs(vector_angle(x) - angle_0))
        else: 
            angle = vector_angle(vector)

            for i, prime in enumerate(close_primes): 
                prime_angle = vector_angle(prime)

                if abs(angle - angle_0) < abs(prime_angle - angle_0): 
                    close_primes = close_primes[0:i] + [vector] + close_primes[i:-1]
                    break

    plot_angles(close_primes)    
    plot_angles([composite], colour="hotpink")         

    close_primes = sorted(close_primes, key=lambda x: abs(EUCLID_NORM(x) - EUCLID_NORM(composite)))   
    plot_angles([close_primes[0]], colour="limegreen")
    print(close_primes[0])
    plt.show()
        
    #[(81+34j), (82+35j), (117+50j), (102+43j), (109+46j), (112+47j), (40+17j), (26+11j), (61+26j), (113+48j), (85+36j), (47+20j)]
    # plot_angles([composite], "hotpink")
    return close_primes[0]


    # p, q = farey.get_pq(min_vector)
    # if p_neg < 0: 
    #     p = -1 * p
    # if q_neg < 0: 
    #     q = -1 * q

    # return farey.farey(p, q)



# base angle set reconstructions -----------------------------------------------
MRI_RECON = 1
CT_RECON = 0
SNR_CT = 0.95
SNR_MRI = 40

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
        num_angles = num_angles_mri
        octant = OCTANT_MRI
        recon = recon_MRI
        path_head = "results_MRI/"
        title = "MRI reconstruction"
    else: #CT RECON
        num_angles = num_angles_ct
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


def comp_recplacement_recon(p, num_angles_octant, iterations, recon_type=MRI_RECON, colour="limegreen", line="-", noisy=False):
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
                prime = closest_gaussian_prime(p, angle)
                subset_angles[i][j] = prime
                comp_replacements.append(prime)

    angles = [angle for subset in subset_angles for angle in subset]
    
    recon_im, rmses, psnrs, ssims = recon(p, angles, remove_empty(subset_angles), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour=colour, line="--", label="prime replacement, " + str(num_angles_ct) + " projections")

    return angles, recon_im, rmses, psnrs, ssims


# reconstructions --------------------------------------------------------------
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
    angles, recon, rmses, psnrs, ssims = regular_recon(p, num_angles_octant, iterations, recon_type=CT_RECON, colour="hotpink", line="-", noisy=noisy)

    # data["regular"] = {"angles": angles, "its": iterations, "rmse": rmses, "psnr": psnrs, "ssim": ssims}
    # path = path_head + "recon_0/its_" + str(iterations)+"_angles_" + str(num_angles_octant) + ".npz"
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

    path_head = "results_MRI/" if recon_type else "results_CT/"
    path = path_head + "recon_1/oct_angles_" + str(num_angles_octant) + "_its_" + str(iterations) + ".npz"
    np.savez(file=path, data=data)


def recon_2(p, num_angles_octant, iterations, noisy=False): 
    data_CT = {}
    data_MRI = {}

    #angles for top two quadrants
    num_angles_MRI = 2 * num_angles_octant - 1  
    num_angles_CT = 2 * (num_angles_MRI - 2)

    angles_CT, subset_angles_CT = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=num_angles_CT) 
    angles_MRI, subset_angles_MRI = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_MRI,K=K, max_angles=num_angles_MRI)  
    
    composites, subset_composites = get_composites(subset_angles_CT)
    
    #set up dictionaries pre recon 
    for [comp] in get_compostie_sets(composites, 0): 
        data_CT[comp] = {}
        data_MRI[comp] = {}

    #reconstruct CT and MRI for the three different equivalent classes 
    for equiv_class_type in [1, 2, 4]: 
        equiv_comp_classes = get_compostie_sets(composites, equiv_class_type)

        plt.figure(figsize=(16, 8))
        colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(2 * equiv_comp_classes)+3)))

        for i, comp_equiv_class in enumerate(equiv_comp_classes):
            
            #reset angle set to prime angles
            new_angles_CT, new_subset_angles_CT = get_primes(subset_angles_CT) #this is stupid it won't seperate the lists even if i list(prime_subset_CT) >:(
            new_angles_MRI, new_subset_angles_MRI = get_primes(subset_angles_MRI)
            
            #add equivalent class to prime angle set 
            for i, angle in enumerate(comp_equiv_class):
                #add at same index as in regular subset
                idx = get_subset_index(angle, subset_composites)
                new_subset_angles_CT[idx].append(angle)
                new_angles_CT.append(angle)

                #only adding first quadrant angles to MRI subset
                a, b = farey.get_pq(angle)
                if a >= 0 and b >=0: 
                    angles_MRI.append(angle)
                    new_subset_angles_MRI[idx].append(angle)
                    new_angles_MRI.append(angle)

            #reconstruct with new compositie + prime subset 
            recon_CT, rmses_CT, psnrs_CT, ssims_CT = recon_CT(p, new_angles_CT, remove_empty(new_subset_angles_CT), iterations, noisy)
            plot_recon(rmses_CT, psnrs_CT, ssims_CT, label=str(comp_equiv_class)+" CT", colour=next(colour))

            recon_MRI, rmses_MRI, psnrs_MRI, ssims_MRI = recon_MRI(p, new_angles_MRI, remove_empty(new_subset_angles_MRI), iterations, noisy)
            plot_recon(rmses_MRI, psnrs_MRI, ssims_MRI, label=str(comp_equiv_class)+" MRI", colour=next(colour))

            #convert to correct key for equiv angle 
            a, b = farey.get_pq(comp_equiv_class[0])
            a, b = min(abs(a), abs(b)), max(abs(a), abs(b))
            comp = farey.farey(a, b)
            data_CT[comp][equiv_class_type] = {"angles": new_angles_CT, "recon":recon_CT, "rmse": rmses_CT, "psnr": psnrs_CT, "ssim": ssims_CT, "noise":noisy}
            data_MRI[comp][equiv_class_type] = {"angles": new_angles_MRI, "recon":recon_MRI, "rmse": rmses_MRI, "psnr": psnrs_MRI, "ssim": ssims_MRI, "noise":noisy}
            
    path = "results_CT/recon_2/oct_angles_" + str(num_angles_octant) + "_its_" + str(iterations) + "_rev2.npz"
    np.savez(file=path, data_CT=data_CT)
    path = "results_MRI/recon_2/oct_angles_" + str(num_angles_octant) + "_its_" + str(iterations) + "_rev2.npz"
    np.savez(file=path, data_MRI=data_MRI)

    plt.savefig("result_MRI_CT/recon_2/angles_" + str(num_angles_octant) + "_its_" + str(iterations) + "_rev2.png")


def recon_3(p, num_angles_octant, iterations, recon_type=MRI_RECON, noisy=False):
    """CT reconstruction for regular angle set, and angle set with all comopsite 
    angles replaced with prime angles. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
    """
    data = {}

    angles_reg, recon, rmses, psnrs, ssims = regular_recon(p, num_angles_octant, iterations, recon_type, noisy=noisy)
    data["regular"] = {"angles": angles_reg, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    angles_rep, recon, rmses, psnrs, ssims = comp_recplacement_recon(p, num_angles_octant, iterations, recon_type, colour="skyblue", noisy=noisy)
    data["prime_replacement"] = {"angles": angles_rep, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}


    plt.figure(figsize=(16, 8))
    plot_angles(angles_reg, colour="hotpink", label="regular")
    plot_angles(angles_rep, colour="skyblue", label="regular", line="--")

    # #prime replacement recon
    # angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=num_angles_CT) 

    # comp_replacements = []
    # for i, subset in enumerate(subset_angles): 
    #     for j, angle in enumerate(subset): 
    #         if not (farey.is_gauss_prime(angle) or abs(angle) == 1): 
    #             prime = closest_gaussian_prime(p, angle)
    #             subset_angles[i][j] = prime
    #             comp_replacements.append(prime)

    # angles = [angle for subset in subset_angles for angle in subset]
    
    # recon, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subset_angles), iterations, noisy)
    # plot_recon(rmses, psnrs, ssims, colour="skyblue", line="--", label="prime replacement")
    # data["prime_replacement"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    # path = "results_ct/recon_3/oct_angles_" + str(num_angles_octant) + "_its_" + str(iterations) + ".npz"
    # np.savez(file=path, data=data)

    
    
def recon_4(p, num_angles_octant, iterations): 
    """Reconstructs MRI and CT with regular angle set but with noise in kSpace 
    and mojette projections. Currenlty, CT recon explodes after starting good. 
    Unsrue why. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
    """
    num_angles_mri = num_angles_octant * OCTANT_MRI - 1 #must be odd to account for (1, 1) having no mirror pair  
    num_angles_ct = 2 * (num_angles_mri - 2) #each angle maps to itself and its vertical mirror EXCEPT (0, 1) and (1, 0) 

    angles_MRI, subset_angles_MRI = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_MRI,K=K, max_angles=num_angles_mri)  
    angles_CT, subset_angles_CT = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=num_angles_ct) 

    # recon, mses, psnrs, ssims = recon_CT(p, angles_CT, remove_empty(subsetAngles_CT), iterations, addNoise = False)
    # plot_recon(mses, psnrs, ssims, colour="hotpink", label="Regular")

    # recon, mses, psnrs, ssims = recon_MRI(p, angles_MRI, remove_empty(subset_angles_MRI), iterations, addNoise = True)
    # plot_recon(mses, psnrs, ssims, colour="skyblue", label="Noisy")
    
    recon, mses, psnrs, ssims = recon_CT(p, angles_CT, remove_empty(subset_angles_CT), iterations, addNoise = True)
    plot_recon(mses, psnrs, ssims, colour="skyblue", label="Noisy CT")

    plt.show()


# additional recons ------------------------------------------------------------
def recon_1b(p, iterations, recon_type=MRI_RECON, noisy=False): 
    data = {"regular":{}, "prime":{}}
    octant_angles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    colour = iter(plt.cm.gist_rainbow(np.linspace(0,1, 2 * len(octant_angles) + 1)))

    for num_angles in octant_angles: 
        angles, rmses, psnrs, ssims = regular_recon(p, num_angles, iterations, recon_type, colour=next(colour), noisy=noisy)
        data["regular"][num_angles] = {"angles": angles, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

        angles, rmses, psnrs, ssims = prime_recon(p, num_angles, iterations, recon_type, colour=next(colour), noisy=noisy)
        data["prime"][num_angles] = {"angles": angles, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    path = "results_ct/recon_1/recon_1b/many_angle_its_" + str(iterations) + ".npz"
    np.savez(file=path, data=data)

    


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


#Shes a runner shes a track star -----------------------------------------------
#recon constants
NUM_OCTANT_ANGLES = 20 + 1
ITERATIONS = 200
OCTANT_MRI = 2
OCTANT_CT = 4
#plotting constants
LINE_MRI = "--"
LINE_CT = '-'
MRI_RECON = 1
CT_RECON = 0

# noise from shakes ------------------------------------------------------------
def add_noise(mt_projs, SNR=0.95):
    '''
    Return (Gaussian) noise of DRT bins as Normal(bins[j],SNR*bins[j]).
    You can then multiply or add this to the bins. Noise is not quantised, do it yourself with astype(np.int32)
    '''
    for m, proj in enumerate(mt_projs):
        for t, bin in enumerate(proj):
            mt_projs[m][t] = random.normalvariate(bin, 0.15*(1.0-SNR)*bin)





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

def createFractal(lines, p, plot=True, ax=plt, title="Fractal"): 
    maxLines = len(lines)   
    color=iter(plt.cm.jet(np.linspace(0,1,maxLines+1)))
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
            print("title")
    return image

def plotFractal(angles, plotReg=True, plotColour=True, ax=plt, title="fractal"): 
    (lines, mValues) = calcFiniteLines(angles)
    fractal = createFractal(lines, p, plot=plotColour, ax=ax, title=title)
    if plotReg: 
        if plotColour: 
            plt.figure()
        plt.imshow(fractal)
    return fractal



            
if __name__ == "__main__": 
    p = nt.nearestPrime(N)
    angle = farey.farey(3, 7)
    angle_2 = closest_gaussian_prime(p, angle)    # recon_3(p, NUM_OCTANT_ANGLES, ITERATIONS, MRI_RECON)
    # plt.show()
    plt.figure()
    plotFractal([angle], plotColour=False)
    plt.figure()
    plotFractal([angle_2], plotColour=False)

    # plt.show()
# %%
