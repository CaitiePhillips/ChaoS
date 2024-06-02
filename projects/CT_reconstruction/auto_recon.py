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


from skimage.io import imread
from skimage import data_dir
import skimage as sk 
from skimage import transform as tfm






# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()


#parameter sets (K, k, i, s, h)
#phantom
# parameters = [1.2, 1, 381, 30, 8.0] #r=2
# parameters = [0.4, 1, 100, 12, 12.0] #r=4
parameters = [1, 1, 100, 30, 12.0] #r=4

#cameraman
# parameters = [1, 1, 381, 30, 8.0] #r=2

#parameters
n = 256 
k = parameters[1]
M = int(k*n)
N = n 

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
smoothIncrement = 5
smoothMaxIteration = iterations/2
relaxIterationFactor = int(0.01*iterations)
#smoothMaxIteration2 = iterations-1
smoothMaxIteration2 = iterations-relaxIterationFactor*smoothIncrement
print("N:", N, "M:", M, "s:", s, "i:", iterations)

# consts -----------------------------------------------------------------------
MRI_RECON = 1
CT_RECON = 0
SNR_CT = 0.95
SNR_MRI = 25.5751
ITERATIONS = 400
FIRST_QUAD = 2
KATZ_ANGLES = -1
VMIN = 0
VMAX = 10
PRIME = "skyblue"
REGULAR = "hotpink"
COMPOSITE = "mediumpurple"
FBP = "limegreen"
OTHER = "cornflowerblue"



def equiv_class(angle, class_type): 
    p, q = farey.get_pq(angle)
    p, q = min(abs(p), abs(q)), max(abs(p), abs(q))
    if class_type == 0: 
        angles = [farey.farey(p, q)]
    elif class_type == 1: 
        angles = [farey.farey(p, q)]#type 1 - 1st and 4th octant
    elif class_type == 2: 
        angles = [farey.farey(q, p)] #type 2 - 2nd and 3rd octant
    elif class_type == 4: 
        angles = [farey.farey(p, q), farey.farey(q, p)] #type 4 - all octants
    return angles


def get_compostie_sets(composites, class_type, recon_type=MRI_RECON): 
    """
    3 class types: 
        type 1: [q + pj, -q + pj] -> return equivalent angle in first oct and vertically mirrored
        type 2: [p + qj, -p + qj] -> return equivalent angle in second oct and vertically mirrored
        type 4: [q + pj,  p + qj, -p + qj, -q + pj] -> both
    """
    subsets = []
    for composite in composites: 
        if composite in [angle for subset in subsets for angle in subset]: 
            continue
        angles = equiv_class(composite, class_type)
        
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


def get_path(recon_type, recon_num, num_angles, iterations, noisy, snr = -1):
    path = "results_" + ("MRI" if recon_type else "CT") + "/"
    path += "recon_" + str(recon_num) + "/"
    path += "num_angles_" + str(num_angles) + "_"
    path += "its_" + str(iterations) + "_"
    path += "noise_" + str(noisy) 
    if snr > 0: 
        path += "_" + str(snr) 
    path += ".npz"
    return path 

# angle helpers ----------------------------------------------------------------
def extend_quadrant(angles_subsets): 
    new_angles = []
    new_angles_subset = []

    for angles in angles_subsets: 
        new_subset = []

        for angle_quad_1 in angles: 
            p, q = farey.get_pq(angle_quad_1)
            if angle_quad_1 not in new_angles: #check only because unsure of how (1, 0) is handeled
                new_angles.append(angle_quad_1)
                new_subset.append(angle_quad_1)
            
            angle_quad_2 = farey.farey(p, -1 * q)
            if angle_quad_2 not in new_angles and abs(angle_quad_1) != 1: 
                new_angles.append(angle_quad_2)
                new_subset.append(angle_quad_2)

        new_angles_subset.append(new_subset)

    return new_angles, new_angles_subset


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
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax = plt
        ax.xlim((-1, 1))
        ax.ylim((0, 1))
        plt.xticks([])
        plt.yticks([])

    for angle in angles: 
        imag, real = farey.get_pq(angle)
        ax.plot([0, real], [0, imag], line, c=colour, linewidth=linewidth, label=label)

# FBP --------------------------------------------------------------------------
def fbp(p, num_angles, noisy=False, snr=SNR_CT):
    image = imread(sk.data_dir + "/phantom.png", as_grey=True)
    image = tfm.rescale(image, scale = float(p) / 400, mode='constant')
    theta = np.linspace(0., 180., num_angles, endpoint=True)
    sinogram = tfm.radon(image, theta=theta, circle=True)

    if noisy: 
        add_noise(sinogram, snr=snr)

    data = {}

    reconstruction_fbp = tfm.iradon(sinogram, theta=theta, circle=True)
    rmse_fbp = np.sqrt(imageio.immse(image, reconstruction_fbp))
    psnr_fbp = imageio.impsnr(image, reconstruction_fbp)
    ssim_fbp = imageio.imssim(image.astype(float), reconstruction_fbp.astype(float))
    data["fbp"] = {"rmse":rmse_fbp, "psnr":psnr_fbp, "ssim":ssim_fbp}

    # reconstruction_sart = tfm.iradon_sart(sinogram, theta=theta)
    # rmse_sart_1 = np.sqrt(imageio.immse(image, reconstruction_sart))
    # psnr_sart_1 = imageio.impsnr(image, reconstruction_sart)
    # ssim_sart_1 = imageio.imssim(image.astype(float), reconstruction_sart.astype(float))
    # data["sart1"] = {"rmse":rmse_sart_1, "psnr":psnr_sart_1, "ssim":ssim_sart_1}


    # reconstruction_sart2 = tfm.iradon_sart(sinogram, theta=theta,
    #                                 image=reconstruction_sart)
    # rmse_sart_2 = np.sqrt(imageio.immse(image, reconstruction_sart2))
    # psnr_sart_2 = imageio.impsnr(image, reconstruction_sart2)
    # ssim_sart_2 = imageio.imssim(image.astype(float), reconstruction_sart2.astype(float))
    # data["sart2"] = {"rmse":rmse_sart_2, "psnr":psnr_sart_2, "ssim":ssim_sart_2}

    return reconstruction_fbp, rmse_fbp, psnr_fbp, ssim_fbp


def fbp_sart(p, num_angles, noisy=False, snr=SNR_CT):
    image = imread(sk.data_dir + "/phantom.png", as_grey=True)
    image = tfm.rescale(image, scale = float(p) / 400, mode='constant')
    theta = np.linspace(0., 180., num_angles, endpoint=True)
    sinogram = tfm.radon(image, theta=theta, circle=True)

    if noisy: 
        add_noise(sinogram, snr=snr)

    data = {}

    reconstruction_fbp = tfm.iradon(sinogram, theta=theta, circle=True)
    rmse_fbp = np.sqrt(imageio.immse(image, reconstruction_fbp))
    psnr_fbp = imageio.impsnr(image, reconstruction_fbp)
    ssim_fbp = imageio.imssim(image.astype(float), reconstruction_fbp.astype(float))
    data["fbp"] = {"rmse":rmse_fbp, "psnr":psnr_fbp, "ssim":ssim_fbp}

    reconstruction_sart = tfm.iradon_sart(sinogram, theta=theta)
    rmse_sart_1 = np.sqrt(imageio.immse(image, reconstruction_sart))
    psnr_sart_1 = imageio.impsnr(image, reconstruction_sart)
    ssim_sart_1 = imageio.imssim(image.astype(float), reconstruction_sart.astype(float))
    data["sart1"] = {"rmse":rmse_sart_1, "psnr":psnr_sart_1, "ssim":ssim_sart_1}


    reconstruction_sart2 = tfm.iradon_sart(sinogram, theta=theta,
                                    image=reconstruction_sart)
    rmse_sart_2 = np.sqrt(imageio.immse(image, reconstruction_sart2))
    psnr_sart_2 = imageio.impsnr(image, reconstruction_sart2)
    ssim_sart_2 = imageio.imssim(image.astype(float), reconstruction_sart2.astype(float))
    data["sart2"] = {"rmse":rmse_sart_2, "psnr":psnr_sart_2, "ssim":ssim_sart_2}

    return reconstruction_sart2, rmse_sart_2, psnr_sart_2, ssim_sart_2


def fbp_sart_w_angles(p, angles, noisy=False, snr=SNR_CT):

    theta = []
    for angle in angles: 
        o, a = farey.get_pq(angle)
        angle = np.rad2deg(np.arctan(float(o)/float(a)))
        if 0 <= angle <= 180:
            if a > 0: 
                theta.append(angle)
                theta.append(180 - angle)
    print(theta)
    theta = np.array(theta)
    image = imread(sk.data_dir + "/phantom.png", as_grey=True)
    image = tfm.rescale(image, scale = float(p) / 400, mode='constant')
    # theta = np.linspace(0., 180., num_angles, endpoint=True)
    sinogram = tfm.radon(image, theta=theta, circle=True)

    if noisy: 
        add_noise(sinogram, snr=snr)

    data = {}

    reconstruction_fbp = tfm.iradon(sinogram, theta=theta, circle=True)
    rmse_fbp = np.sqrt(imageio.immse(image, reconstruction_fbp))
    psnr_fbp = imageio.impsnr(image, reconstruction_fbp)
    ssim_fbp = imageio.imssim(image.astype(float), reconstruction_fbp.astype(float))
    data["fbp"] = {"rmse":rmse_fbp, "psnr":psnr_fbp, "ssim":ssim_fbp}

    reconstruction_sart = tfm.iradon_sart(sinogram, theta=theta)
    rmse_sart_1 = np.sqrt(imageio.immse(image, reconstruction_sart))
    psnr_sart_1 = imageio.impsnr(image, reconstruction_sart)
    ssim_sart_1 = imageio.imssim(image.astype(float), reconstruction_sart.astype(float))
    data["sart1"] = {"rmse":rmse_sart_1, "psnr":psnr_sart_1, "ssim":ssim_sart_1}


    reconstruction_sart2 = tfm.iradon_sart(sinogram, theta=theta,
                                    image=reconstruction_sart)
    rmse_sart_2 = np.sqrt(imageio.immse(image, reconstruction_sart2))
    psnr_sart_2 = imageio.impsnr(image, reconstruction_sart2)
    ssim_sart_2 = imageio.imssim(image.astype(float), reconstruction_sart2.astype(float))
    data["sart2"] = {"rmse":rmse_sart_2, "psnr":psnr_sart_2, "ssim":ssim_sart_2}

    plt.imshow(reconstruction_sart2)

    return reconstruction_sart2, rmse_sart_2, psnr_sart_2, ssim_sart_2



# Base for CT and MRI reconstructions ------------------------------------------
def angleSubSets_Symmetric(s, mode, P, Q, octant=FIRST_QUAD, binLengths=False, K = 1, prime_only = False, max_angles = 10, norm=EUCLID_NORM):
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


def recon_CT(p, angles, subsetAngles, iterations, noisy=False, snr=SNR_CT): 

    lena, mask = imageio.phantom(N, p, True, np.uint32, True)
    
    # lena, mask = imageio.cameraman(N, p, True, np.uint32, True)


    #get half plane of angles
    angles, subsetAngles = extend_quadrant(subsetAngles)

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
        add_noise(mt_lena, snr) 

    rt_lena = mojette.toDRT(mt_lena, angles, p, p, p) 
    
    recon, mses, psnrs, ssims = osem_expand(iterations, p, rt_lena, remove_empty(subsetsMValues), finite.frt, finite.ifrt, lena, mask)
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

    recon, mses, psnrs, ssims = osem_expand_complex(iterations, p, drtSpace, remove_empty(subsetsMValues), finite.frt_complex, finite.ifrt_complex, image, mask)

    return recon, mses, psnrs, ssims


# base angle set reconstructions -----------------------------------------------
def regular_recon(p, angles, subsetAngles, iterations, recon_type=MRI_RECON, colour="hotpink", line="-", noisy=False, s=s, K=K, snr=SNR_CT):
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

    if recon_type == MRI_RECON: #MRI RECON
        recon = recon_MRI
        path_head = "results_MRI/"
        title = "MRI reconstruction" + (" (noisy)" if noisy else "")
    else: #CT RECON
        recon = recon_CT
        path_head = "results_CT/"
        title = "CT reconstruction" + (" (noisy)" if noisy else "")

    # angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=num_angles)  
    # if recon_type == CT_RECON: 
    #     angles, subsetAngles = extend_quadrant(subsetAngles)

    recon_im, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, noisy, snr=snr)
    # plot_recon(rmses, psnrs, ssims, colour=colour, line=line, label="regular recon, " + str(num_angles_ct) + " projections") #normal
    # plot_recon(rmses, psnrs, ssims, colour=colour, line=line, label=title + ", " + str(len(angles)) + " projections")

    # plt.suptitle(title)

    return angles, recon_im, rmses, psnrs, ssims


def prime_recon(p, angles, subset_angles, iterations, recon_type=MRI_RECON, colour="skyblue", line="-", noisy=False, snr=SNR_CT): 
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

    # angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=num_angles) 
    primes, primes_subset = get_primes(subset_angles)
    recon_im, rmses, psnrs, ssims = recon_CT(p, primes, remove_empty(primes_subset), iterations, noisy, snr=snr)
    # plot_recon(rmses, psnrs, ssims, colour=colour, line=line, label="Prime subset, " + str(len(primes)) + " projections")

    return primes, recon_im, rmses, psnrs, ssims


def composite_recon(p, angles, subset_angles, iterations, recon_type=MRI_RECON, colour="mediumpurple", line="-", noisy=False, snr=SNR_CT): 
    """Completes one MRI or CT reconstruction with only the composite angle set. 
    Plots error info. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon (int, optional): Specificy MRI recon (=1) or CT recon (=0). 
        Defaults to 0
    """

    if recon_type: #MRI RECON
        recon = recon_MRI
        title = "MRI composite reconstruction" + (" (noisy)" if noisy else "")
    else: #CT RECON
        recon = recon_CT
        title = "CT composite reconstruction" + (" (noisy)" if noisy else "")

    # angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=num_angles) 
    comps, comps_subset = get_composites(subset_angles)
    recon_im, rmses, psnrs, ssims = recon_CT(p, comps, remove_empty(comps_subset), iterations, noisy, snr=snr)
    plot_recon(rmses, psnrs, ssims, colour=colour, line=line, label="Composite subset, " + str(len(comps)) + " projections")

    return comps, recon_im, rmses, psnrs, ssims


def comp_recplacement_recon(p, num_angles, iterations, num_to_store=20, recon_type=MRI_RECON, colour="limegreen", line="-", noisy=False):
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
    if recon_type: #MRI RECON
        recon = recon_MRI
        title = "MRI composite replacement reconstruction" + (" (noisy) " if noisy else " ")
    else: #CT RECON
        recon = recon_CT
        title = "CT composite replacement reconstruction" + (" (noisy) " if noisy else " ")


    
    #prime replacement recon
    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K,max_angles=num_angles) 

    comp_replacements = []
    for i, subset in enumerate(subset_angles): 
        for j, angle in enumerate(subset): 
            if not (farey.is_gauss_prime(angle) or abs(angle) == 1): 
                prime = closest_gaussian_prime(p, angle, num_to_store)
                subset_angles[i][j] = prime
                comp_replacements.append(prime)

    angles = [angle for subset in subset_angles for angle in subset]
    
    recon_im, rmses, psnrs, ssims = recon(p, angles, remove_empty(subset_angles), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour=colour, label=title + str(len(angles)) + " projections")
    
    return angles, recon_im, rmses, psnrs, ssims


#fractal functions -------------------------------------------------------------
def calcFiniteLines(angles, p): 
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


def createFractal(lines, p, file_name, ax): 
    powerSpect = np.zeros((p,p))
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    plt.gray()
    plt.tight_layout()

    maxLines = len(lines)
    #maxLines = 12
    # ax[0].imshow(powerSpect)
    # ax[1].imshow(powerSpect)
    # color=iter(cm.rainbow(np.linspace(0,1,len(lines))))
    # color=iter(plt.cm.jet(np.linspace(0,1,maxLines+1)))
    fareyImage = np.zeros_like(powerSpect)
    for i, line in enumerate(lines):
        u, v = line
        # c=next(color)
        # ax[0].plot(u, v, '.', c=c)
        # ax[1].plot(u, v, '.r',markersize=1)
        fareyImage[u,v] = 255
        if i == maxLines:
            break
    fareyImage[125,125] = 0
    ax.imshow(fareyImage)
    # ax[0].set_title('Sampling (colour per line) for prime size:'+str(p))
    # ax[1].set_title('Sampling (same colour per line) for prime size:'+str(p))
    # imageio.imsave("results_CT/recon_1c/" + file_name + ".png", fareyImage)


def plotFractal(p, angles, recon_type, ax, title="fractal", num_to_store=0): 
    lines, mValues = calcFiniteLines(angles, p)
    createFractal(lines, p, title, ax)
    # path = "fraC.png"
    # plt.savefig(path)


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
    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=num_angles) 
    
    recon_im, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, False)
    data["no noise"] = {"recon": recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims}
    # plot_recon(rmses, psnrs, ssims, colour="hotpink", line="-", label="not noisy recon, " + str(num_angles) + " projections")

    recon_im_noisy, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, True)
    # plot_recon(rmses, psnrs, ssims, colour="skyblue", line="--", label="noisy recon, " + str(num_angles) + " projections")
    data["noise"] = {"recon": recon_im_noisy, "rmse": rmses, "psnr": psnrs, "ssim": ssims}

    angles, subsetAngles = extend_quadrant(subsetAngles)
    recon_im_fbp, rmse, psnr, ssim = fbp(p, len(angles))
    rmses = rmse * np.ones_like(rmses)
    psnrs = psnr * np.ones_like(psnrs)
    ssims = ssim * np.ones_like(ssims)
    data["FBP no noise"] = {"recon": recon_im_fbp, "rmse": rmses, "psnr": psnrs, "ssim": ssims}

    recon_im_fbp_noisy, rmse, psnr, ssim = fbp(p, len(angles), noisy=True)
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
    recon = recon_CT
    path_head = "results_CT/"
    title = "CT reconstruction"

    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=num_angles)  

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


def recon_1(p, iterations, num_angles=KATZ_ANGLES, recon_type=CT_RECON, noisy=False, colours=["hotpink", "skyblue", "mediumpurple"]): 
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
    line = "--" if noisy else "-" 

    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES) 
    
    reg_angles, recon, rmses, psnrs, ssims = regular_recon(p, angles, subset_angles, iterations, recon_type, noisy=noisy, line=line, colour=colours[0])
    data["regular"] = {"angles": reg_angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    prime_angles, recon, rmses, psnrs, ssims = prime_recon(p, angles, subset_angles, iterations, recon_type, noisy=noisy, line=line, colour=colours[1])
    data["prime"] = {"angles": prime_angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    comp_angles, recon, rmses, psnrs, ssims = composite_recon(p, angles, subset_angles, iterations, recon_type, noisy=noisy, line=line, colour=colours[2])
    data["composite"] = {"angles": comp_angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    recon_im_fbp, rmse, psnr, ssim = fbp(p, len(reg_angles))
    rmses = rmse * np.ones_like(rmses)
    psnrs = psnr * np.ones_like(psnrs)
    ssims = ssim * np.ones_like(ssims)
    data["FBP"] = {"angles": reg_angles, "recon": recon_im_fbp, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    # plotFractal(reg_angles, CT_RECON, "regular")
    # plotFractal(prime_angles, CT_RECON, "prime")
    # plotFractal(comp_angles, CT_RECON, "composite")

    path = get_path(recon_type, 1, num_angles, iterations, noisy)
    np.savez(file=path, data=data)


def recon_2(p, num_angles, iterations, recon_type=MRI_RECON, noisy=False): 

    if recon_type: #MRI RECON
        recon = recon_MRI
        title = "MRI reconstruction"
    else: #CT RECON
        recon = recon_CT
        title = "CT reconstruction"

    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES) 
    composites, subset_composites = get_composites(subset_angles)
    
    #set up dictionary pre recon 
    data = {}

    #base line 
    primes, recon_im, rmses, psnrs, ssims = prime_recon(p, angles, subset_angles, iterations, recon_type, noisy=noisy, colour="hotpink")
    data["prime"] = {"angles": primes, "recon": recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    angles, recon_im, rmses, psnrs, ssims = regular_recon(p, angles, subset_angles, iterations, recon_type, noisy=noisy, colour="skyblue")
    data["regular"] = {"angles": angles, "recon": recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
 
    for [comp] in get_compostie_sets(composites, 0): 
        data[comp] = {}

    #reconstruct CT and MRI for the three different equivalent classes 
    # for equiv_class_type in [1, 2, 4]: 
    equiv_class_type = 4
    equiv_comp_classes = get_compostie_sets(composites, equiv_class_type, recon_type)

    plt.figure(figsize=(16, 8))
    plt.suptitle(title)
    colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(equiv_comp_classes)+3)))

    for i, comp_equiv_class in enumerate(equiv_comp_classes):
        print(comp_equiv_class)
        
        #reset angle set to prime angles
        new_angles, new_subset_angles = get_primes(subset_angles) 
        
        #add composite angles to prime angle set 
        for i, angle in enumerate(comp_equiv_class):
            #add at same index as in regular subset
            idx = get_subset_index(angle, subset_composites)
            new_subset_angles[idx].append(angle)
            new_angles.append(angle)

        #reconstruct with new compositie + prime subset 
        recon_im, rmses, psnrs, ssims = recon_CT(p, new_angles, remove_empty(new_subset_angles), iterations, noisy)
        plot_recon(rmses, psnrs, ssims, label=str(comp_equiv_class), colour=next(colour))

        #use first octant as key 
        a, b = farey.get_pq(comp_equiv_class[0])
        comp = farey.farey(min(abs(a), abs(b)), max(abs(a), abs(b)))
        print(comp)
        data[comp][equiv_class_type] = {"angles": new_angles, "recon":recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
    
    path = get_path(recon_type, 2, num_angles, iterations, noisy)
    np.savez(file=path, data=data)

    # plt.savefig("result_MRI_CT/recon_2/angles_" + str(num_angles_octant) + "_its_" + str(iterations) + "_rev2.png")


def recon_2b(p, num_angles, iterations, recon_type=MRI_RECON, noisy=False): 

    if recon_type: #MRI RECON
        recon = recon_MRI
        title = "MRI reconstruction"
    else: #CT RECON
        recon = recon_CT
        title = "CT reconstruction"

    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES) 
    composites, subset_composites = get_composites(subset_angles)
    
    #set up dictionary pre recon 
    data = {}

    #base line 
    recon_im, rmses, psnrs, ssims = recon_CT(p, angles, subset_angles, iterations, noisy=noisy)
    data["regular"] = {"angles": angles, "recon": recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    primes, subset_primes = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES, prime_only=True) 
    recon_im, rmses, psnrs, ssims = recon_CT(p, primes, subset_primes, iterations, noisy=noisy)
    data["prime"] = {"angles": primes, "recon": recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
 
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
            new_angles = list(primes)
            new_subset_angles = list(subset_primes)
            
            #add composite angles to prime angle set 
            for i, angle in enumerate(comp_equiv_class):
                #add at same index as in regular subset
                idx = get_subset_index(angle, subset_composites)
                new_subset_angles[idx].append(angle)
                new_angles.append(angle)

            #reconstruct with new compositie + prime subset 
            recon_im, rmses, psnrs, ssims = recon_CT(p, new_angles, remove_empty(new_subset_angles), iterations, noisy)
            # plot_recon(rmses, psnrs, ssims, label=str(comp_equiv_class), colour=next(colour))

            #use first octant as key 
            a, b = farey.get_pq(comp_equiv_class[0])
            comp = farey.farey(min(abs(a), abs(b)), max(abs(a), abs(b)))
            data[comp][equiv_class_type] = {"angles": new_angles, "recon":recon_im, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
    
    path = get_path(recon_type, "2b", num_angles, iterations, noisy)
    np.savez(file=path, data=data)

    # plt.savefig("result_MRI_CT/recon_2/angles_" + str(num_angles_octant) + "_its_" + str(iterations) + "_rev2.png")


def recon_3(p, num_angles, iterations, recon_type=MRI_RECON, noisy=False):
    """CT reconstruction for regular angle set, and angle set with all comopsite 
    angles replaced with prime angles. 

    Args:
        p (int): prime size of image
        num_angles (int): number of angles per octant
        iterations (int): number of OSEM iterations
        recon_type (int, optional): Selects MRI or CT reconstruction. Defaults to MRI_RECON.
        noisy (bool, optional): Selects to add noise to reconstruction. Defaults to False.
    """
    colour = iter(plt.cm.gist_rainbow(np.linspace(0,1, 5 + 1)))
    data = {}

    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=num_angles)  
    if recon_type == CT_RECON: 
        angles, subsetAngles = extend_quadrant(subsetAngles)

    angles, recon, rmses, psnrs, ssims = regular_recon(p, angles, subsetAngles, iterations, recon_type, noisy=noisy)
    data["regular"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

    angles_rep, recon, rmses, psnrs, ssims = comp_recplacement_recon(p, num_angles, iterations, num_to_store=20, recon_type=recon_type, colour="skyblue", noisy=noisy)
    data["prime_replacement"] = {"angles": angles_rep, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
    
    # data["prime_replacement"] = {}
    # for num_to_store in [5, 10, 20, 30]: 
    #     angles, recon, rmses, psnrs, ssims = comp_recplacement_recon(p, num_angles, iterations, num_to_store=num_to_store, recon_type=recon_type, colour="skyblue", noisy=noisy)
    #     data["prime_replacement"][num_to_store] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}
    
    plt.figure(figsize=(16, 8))
    plot_angles(angles, colour="hotpink")
    plot_angles(angles_rep, colour="skyblue", line="--")

    path = get_path(recon_type, 3, num_angles, iterations, noisy)
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


def recon_1c(p, iterations, recon_type=CT_RECON, noisy=False, snr=SNR_CT):
    # title = "CT prime reconstruction" + (" (noisy)" if noisy else "")
    data = {}

    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES)  
    # angles, subsetAngles = extend_quadrant(subsetAngles)

    #normal prime recon
    angles_reg, recon, rmses, psnrs, ssims = regular_recon(p, list(angles), list(subsetAngles), iterations, recon_type, noisy=noisy, colour="hotpink", snr=snr)
    data["regular"] = {"angles": angles_reg, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy, "SNR": snr}
    angles_prime_subset, recon, rmses, psnrs, ssims = prime_recon(p, list(angles), list(subsetAngles), iterations, recon_type, noisy=noisy, colour="skyblue", snr=snr)
    data["prime_subset"] = {"angles": angles_prime_subset, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy, "SNR": snr}

    #prime recon satisfying Kazt 
    primes, primes_subsets = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES, prime_only=True) 
    recon, rmses, psnrs, ssims = recon_CT(p, primes, remove_empty(primes_subsets), iterations, noisy, snr=snr)
    data["prime_katz"] = {"angles": primes, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy, "SNR": snr}
    # plot_recon(rmses, psnrs, ssims, colour="mediumpurple", label="Prime angle set " + str(len(primes)) + " projections")

    # #rergular with same length as prime katz
    # angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=len(primes))  
    # angles, recon, rmses, psnrs, ssims = regular_recon(p, angles, subsetAngles, iterations, recon_type, noisy=noisy, colour="orange", snr=snr)
    # data["regular_prime_len"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy, "SNR": snr}
    


    #fbp
    recon_im_fbp_noisy, rmse, psnr, ssim = fbp(p, len(primes), noisy=noisy, snr=snr)
    rmses = rmse * np.ones_like(rmses)
    psnrs = psnr * np.ones_like(psnrs)
    ssims = ssim * np.ones_like(ssims)
    data["FBP"] = {"recon": recon_im_fbp_noisy, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy, "SNR": snr}
    # plot_recon(rmses, psnrs, ssims, colour="limegreen", label="FBP " + str(len(primes)) + " projections")

    #fbp + sart
    recon_im_fbp_sart_noisy, rmse, psnr, ssim = fbp_sart(p, len(primes), noisy=noisy, snr=snr)
    rmses = rmse * np.ones_like(rmses)
    psnrs = psnr * np.ones_like(psnrs)
    ssims = ssim * np.ones_like(ssims)
    data["FBPSART"] = {"recon": recon_im_fbp_sart_noisy, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy, "SNR": snr}
    # plot_recon(rmses, psnrs, ssims, colour="limegreen", label="FBP SART " + str(len(primes)) + " projections")

    snr = snr*100 if noisy else 100
    path = get_path(recon_type, "1c", KATZ_ANGLES, iterations, noisy, snr=int(snr))
    np.savez(file=path, data=data)
    
# plotters ---------------------------------------------------------------------
def plot_recon(rmseValues, psnrValues, ssimValues, axs, colour = "b", line = '-', label="label", max_it = -1):
    [rmse_ax, ssim_ax, psnr_ax] = axs

    incX = np.arange(0, len(rmseValues[:max_it]))*plotIncrement
    # plt.subplot(1, 3, 1)
    rmse_ax.plot(incX, rmseValues[:max_it], c=colour, ls=line, label=label)
    # plt.title('Error Convergence of the Finite OSEM', fontsize=20)
    rmse_ax.set_xlabel('Iterations', fontsize=XY_LABEL_FONT_SIZE)
    rmse_ax.set_ylabel('RMSE', fontsize=XY_LABEL_FONT_SIZE)

    ssim_ax.plot(incX, ssimValues[:max_it], c=colour, ls=line, label=label)
    ssim_ax.set_ylim([0, 1.0])
    # plt.title('Simarlity Convergence of the Finite OSEM', fontsize=20)
    ssim_ax.set_xlabel('Iterations', fontsize=XY_LABEL_FONT_SIZE)
    ssim_ax.set_ylabel('SSIM', fontsize=XY_LABEL_FONT_SIZE)

    psnr_ax.plot(incX, psnrValues[:max_it], c=colour, ls=line, label=label)
    psnr_ax.set_ylim([0, 60])
    # plt.title('PSNR Convergence of the Finite OSEM', fontsize=20)
    psnr_ax.set_xlabel('Iterations', fontsize=XY_LABEL_FONT_SIZE)
    psnr_ax.set_ylabel('PSNR', fontsize=XY_LABEL_FONT_SIZE)
    # plt.legend(fontsize=20)

    
    # plt.tight_layout(rect=[0, 0.1, 1, 0.95])


def plot_neg_2(path, num_angles): 
    data = np.load(path)["data"].item()

    #plot errors
    plt.figure(figsize=(16, 8))
    plot_recon(data["no noise"]["rmse"], data["no noise"]["psnr"], data["no noise"]["ssim"], label="ChoaS no noise", colour=REGULAR)
    plot_recon(data["noise"]["rmse"], data["noise"]["psnr"], data["noise"]["ssim"], label="ChoaS noise", colour=REGULAR, line='--')
    plot_recon(data["FBP no noise"]["rmse"], data["FBP no noise"]["psnr"], data["FBP no noise"]["ssim"], label="FBP no noise", colour=FBP)
    plot_recon(data["FBP noise"]["rmse"], data["FBP noise"]["psnr"], data["FBP noise"]["ssim"], label="FBP noise", colour=FBP, line='--')
    plt.suptitle("num projs: " + str(num_angles))

    #plot recons
    recon_im = data["no noise"]["recon"]
    recon_im_noisy = data["noise"]["recon"]
    recon_im_fbp = data["FBP no noise"]["recon"]
    recon_im_fbp_noisy = data["FBP noise"]["recon"]

    lena, mask = imageio.phantom(N, p, True, np.uint32, True)
    lena_fbp = imread(data_dir + "/phantom.png", as_grey=True)
    lena_fbp = tfm.rescale(lena_fbp, scale = float(p) / 400, mode='constant')

    fig, axs = plt.subplots(2, 4, figsize=(18, 12))#, sharey=True)
    axs = axs.flat
    axs[0].imshow(recon_im, cmap="gray")
    axs[1].imshow(abs(recon_im - lena), cmap="gray")
    axs[2].imshow(recon_im, cmap="gray")
    axs[3].imshow(abs(recon_im_noisy - lena), cmap="gray")
    axs[4].imshow(recon_im_fbp, cmap="gray")
    axs[5].imshow(abs(recon_im_fbp - lena_fbp), cmap="gray")
    axs[6].imshow(recon_im_fbp_noisy, cmap="gray")
    axs[7].imshow(abs(recon_im_fbp_noisy - lena_fbp), cmap="gray")
    plt.suptitle("num projs: " + str(num_angles))


def plot_recon_1(path, noisy, axs, colours = None): 
    """Plots reg and prime recons for increasing number of angles for octant. 
    """
    # line = "--" if noisy else "-"
    line = '-' if not noisy else "--"
    data = np.load(path)["data"].item()
    reg_recon = data["regular"]
    prime_recon = data["prime"]
    comp_recon = data["composite"]
    fbp_recon = data["FBP"]

    angles, _ = extend_quadrant([reg_recon["angles"]])
    plot_recon(reg_recon["rmse"], reg_recon["psnr"], reg_recon["ssim"], axs,
               colour = REGULAR if colours == None else colours[0], 
               line = line,
               label=r"Regular angle set {}".format("(noise) " if noisy else "") + r" |$\Theta$| = " + str(len(angles)), max_it=400)
    
    angles, _ = extend_quadrant([prime_recon["angles"]])
    plot_recon(prime_recon["rmse"], prime_recon["psnr"], prime_recon["ssim"], axs,
               colour = PRIME if colours == None else colours[1], 
               line = line,
               label="Prime subset {}".format("(noise) " if noisy else "") + r" |$\Theta$| = " + str(len(angles)), max_it=400)
    
    angles, _ = extend_quadrant([comp_recon["angles"]])
    plot_recon(comp_recon["rmse"], comp_recon["psnr"], comp_recon["ssim"], axs,
               colour = COMPOSITE if colours == None else colours[2], 
               line = line,
               label="Composite subset {}".format("(noise) " if noisy else "") + r" |$\Theta$| = " + str(len(angles)), max_it=400)
    
    # plot_recon(fbp_recon["rmse"], fbp_recon["psnr"], fbp_recon["ssim"], 
    #            colour = FBP, 
    #            line = line,
    #            label="FBP {}".format("(noisy) " if noisy else "") + str(len(reg_recon["angles"])) + " projections", max_it=400)
    

def plot_recon_1b(path): 
    """Plots reg and prime recons for increasing number of angles for octant. 
    """
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
    

def plot_recon_1c(path, noisy, axs, colours = None): 
    # line = "--" if noisy else "-"
    line = '-' if not noisy else "--"
    data = np.load(path)["data"].item()
    reg = data["regular"]
    prime_subset = data["prime_subset"]
    prime_kazt = data["prime_katz"]
    # regular_prime_len = data["regular_prime_len"]
    fbp = data["FBP"]
    fbp_sart_recon = data["FBPSART"]
    colour=iter(plt.cm.rainbow(np.linspace(0,1,6)))

    angles, _ = extend_quadrant([reg["angles"]])
    plot_recon(reg["rmse"], reg["psnr"], reg["ssim"], axs, 
               colour = next(colour), 
               line = line,
               label="ChaoS " + ("(noise) " if noisy else "") + r" |$\Theta$| = " + str(len(angles)))
    

    angles, _ = extend_quadrant([prime_subset["angles"]])
    plot_recon(prime_subset["rmse"], prime_subset["psnr"], prime_subset["ssim"], axs, 
               colour = next(colour), 
               line = line,
               label="Prime subset " + ("(noise) " if noisy else "") + r" |$\Theta$| = " + str(len(angles)))
    
    angles, _ = extend_quadrant([prime_kazt["angles"]])
    plot_recon(prime_kazt["rmse"], prime_kazt["psnr"], prime_kazt["ssim"], axs, 
               colour = next(colour) if colours == None else colours[0], 
               line = line,
               label="Prime ChaoS " + ("(noise) " if noisy else "") + r" |$\Theta$| = " + str(len(angles)))#+ r"||$\Theta$|| = " + str(len(prime_kazt["angles"])))


    # plot_recon(regular_prime_len["rmse"], regular_prime_len["psnr"], regular_prime_len["ssim"], 
    #            colour = next(colour), 
    #            line = line,
    #            label="ChaoS Reconstruction {}".format("(noisy) " if noisy else "") + str(len(regular_prime_len["angles"])) + " projections")
    # angles, _ = extend_quadrant([reg["angles"]])
    # plot_recon(fbp["rmse"], fbp["psnr"], fbp["ssim"], axs,
    #            colour = next(colour), 
    #            line = "-",
    #            label="FBP {}".format("(noise) " if noisy else "") + r" |$\Theta$| = " + str(len(angles)))
    # angles, _ = extend_quadrant([reg["angles"]])
    # plot_recon(fbp_sart_recon["rmse"], fbp_sart_recon["psnr"], fbp_sart_recon["ssim"], axs,
    #            colour = next(colour) if colours == None else colours[1], 
    #            line = line,
    #            label="SART {}".format("(noise) " if noisy else "") + r" |$\Theta$| = " + str(len(angles)))#+ str(len(reg["angles"])) + " projections")

    # recon_im_fbp_noisy, rmse, psnr, ssim = fbp(p, 47, noisy=noisy, snr=0.95)
    # rmses = rmse * np.ones_like(prime_kazt["rmse"])
    # psnrs = psnr * np.ones_like(prime_kazt["rmse"])
    # ssims = ssim * np.ones_like(prime_kazt["rmse"])
    # # data["FBP"] = {"recon": recon_im_fbp_noisy, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy, "SNR": snr}
    # plot_recon(rmses, psnrs, ssims, axs, colour=next(colour), 
    #            label="FBP {}".format("(noisy) " if noisy else "") + r"||$\Theta$|| = 57")

    # #fbp + sart
    # recon_im_fbp_sart_noisy, rmse, psnr, ssim = fbp_sart(p, 47, noisy=noisy, snr=0.95)
    # rmses = rmse * np.ones_like(prime_kazt["rmse"])
    # psnrs = psnr * np.ones_like(prime_kazt["rmse"])
    # ssims = ssim * np.ones_like(prime_kazt["rmse"])
    # # data["FBPSART"] = {"recon": recon_im_fbp_sart_noisy, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy, "SNR": snr}
    # plot_recon(rmses, psnrs, ssims, axs, colour=next(colour), 
    #            label="FBP + SART {}".format("(noisy) " if noisy else "") + r"||$\Theta$|| = 57")



    # plotFractal(prime_kazt["angles"], CT_RECON, title="prime_katz")
    # # plt.suptitle("Prime Katz")
    # plotFractal(regular_prime_len["angles"], CT_RECON, title="reg_len_prime_katz")
    # # plt.suptitle("Equivalent length Regular recon")


def plot_recon_2(path, noisy, plot_type=True):
    line = "--" if noisy else "-"

    data = np.load(path)['data'].item()
    equiv_class_types = [1, 2, 4]

    if plot_type:
        for angle in data.keys(): 
            if angle == "regular" or angle == "prime": 
                    continue
            plt.figure(figsize=(16, 8))
            colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(equiv_class_types)+3)))

            #plot base line 
            error_info = data["prime"]
            plot_recon(error_info["rmse"], error_info["psnr"], error_info["ssim"], 
                        label = "Prime reconstruction" + (" (noisy)" if noisy else ""), 
                        colour = next(colour), 
                        line=":")
            print("prime len: " + len(data["prime"]["angles"]))
            
            error_info = data["regular"]
            plot_recon(error_info["rmse"], error_info["psnr"], error_info["ssim"], 
                        label = "Regular reconstruction" + (" (noisy)" if noisy else ""), 
                        colour = next(colour), 
                        line=":")
            print("reg len: " + len(data["prime"]["angles"]))
            
            #plot angle equiv classes
            # for equiv_class_type in equiv_class_types: 
            equiv_class_type = 4
            error_info = data[angle][equiv_class_type]
            plot_recon(error_info["rmse"], error_info["psnr"], error_info["ssim"], 
                        label = str(equiv_class(angle, equiv_class_type)) + (" (noisy)" if noisy else ""), 
                        colour = next(colour), 
                        line=line)

    else: #plot different equiv classes
        # for equiv_class_type in equiv_class_types: 
        equiv_class_type = 4
        plt.figure(figsize=(16, 8))
        colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(data.keys()) + 1)))

        #plot base line 
        error_info = data["prime"]
        plot_recon(error_info["rmse"], error_info["psnr"], error_info["ssim"], 
                    label = "Prime reconstruction" + (" (noisy)" if noisy else ""), 
                    colour = next(colour), 
                    line=":")
        print("prime len: " + str(len(data["prime"]["angles"])))
        error_info = data["regular"]
        plot_recon(error_info["rmse"], error_info["psnr"], error_info["ssim"], 
                    label = "Regular reconstruction" + (" (noisy)" if noisy else ""), 
                    colour = next(colour), 
                    line=":")
        print("reg len: " + str(len(data["regular"]["angles"])))
        
        #plot prime + composite
        for angle in data.keys(): 
            if angle == "regular" or angle == "prime": 
                continue
            error_info = data[angle][equiv_class_type]
            plot_recon(error_info["rmse"], error_info["psnr"], error_info["ssim"], 
                           label = str(equiv_class(angle, equiv_class_type)) + (" (noisy)" if noisy else ""), 
                           colour = next(colour), 
                           line=line)


def plot_recon_3(path, noisy, axs): 
    line = "--" if noisy else "-"

    data = np.load(path)["data"].item()
    reg_recon = data["regular"]
    angle_rep_recon = data["prime_replacement"]

    plot_recon(reg_recon["rmse"], reg_recon["psnr"], reg_recon["ssim"], axs,
               colour = "hotpink", 
               line = line,
               label="Regular reconstruction {}".format("(noisy) " if noisy else ""))
    
    plot_recon(angle_rep_recon["rmse"], angle_rep_recon["psnr"], angle_rep_recon["ssim"], axs,
               colour = "skyblue", 
               line = line,
               label="Composite replacement reconstruction {}".format("(noisy) " if noisy else ""))
    
    
    
    # plt.figure(figsize=(16, 8))
    # angles, _ = extend_quadrant([reg_recon["angles"]])
    # plot_angles(angles, colour = UQ_BLUE)
    # angles, _ = extend_quadrant([angle_rep_recon["angles"]])
    # plot_angles(angles, colour = UQ_RED)


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
    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES)  
    recon_im, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour="hotpink", label="regular recon, " + str(len(angles)) + " projections")

    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES, prime_only=True)  
    recon_im, rmses, psnrs, ssims = recon_CT(p, angles, remove_empty(subsetAngles), iterations, noisy)
    plot_recon(rmses, psnrs, ssims, colour="skyblue", label="prime recon, " + str(len(angles)) + " projections")
    # plt.suptitle(title)


#exps --------------------------------------------------------------------------
def exp_0(run):

    if run: 
        p = nt.nearestPrime(N)
        recon_neg_2(p, ITERATIONS, KATZ_ANGLES)
    else: 
        path = "results_CT/recon_neg_2/FBP_ChaoS_num_angles_{}.npz".format(KATZ_ANGLES)
        plot_neg_2(path, KATZ_ANGLES)

#idk how helpful this one is lol
def exp_1(run): 
    """At the moment idk what is really happening here - this could be evidencie
      of less and more angles than Kazt and how it effects reconstruction. But rn, 
      it doesn't seem like much. 
    """
    if run: 
        p = nt.nearestPrime(N)
        its = 760
        angles, _ = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES) 
        katz_num_angles = len(angles)
        for k in [0.25, 0.5, 0.75, 1]: 
            plt.figure(figsize=(16, 8))
            recon_neg_2(p, its, np.floor(k * katz_num_angles))
    else:
        for k in [0.25, 0.5, 0.75, 1]: #you need to check these angles are making sense, also probs do more 0.9 - 1.1 angles and see how it changes 
            path = "results_CT/recon_neg_2/FBP_ChaoS_num_angles_{}.npz".format(k * katz_num_angles)
            # plot_neg_2(path, k * katz_num_angles)
            data = np.load(path)["data"].item()
            plot_recon(data["no noise"]["rmse"], data["no noise"]["psnr"], data["no noise"]["ssim"], label="ChoaS no noise", colour="hotpink")
            plot_recon(data["noise"]["rmse"], data["noise"]["psnr"], data["noise"]["ssim"], label="ChoaS noise", colour="hotpink", line='--')
            plot_recon(data["FBP no noise"]["rmse"], data["FBP no noise"]["psnr"], data["FBP no noise"]["ssim"], label="FBP no noise", colour="mediumpurple")
            plot_recon(data["FBP noise"]["rmse"], data["FBP noise"]["psnr"], data["FBP noise"]["ssim"], label="FBP noise", colour="mediumpurple", line='--')
            # plt.suptitle("num projs: " + str(num_angles))


def exp_2(run): 
    """Evidence importance of prime numbers in reconstruction as compared to 
    composite. 
    Run for both no noise and noise. 

    Want to find when prime 
    """
    p = nt.nearestPrime(N)
    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES) 
    katz_len = len(angles)
    noisy = True
    if run: 
        for c in [0.5, 1, 2, 4]: 
            for noisy in (True, False): 
                plt.figure(figsize=(16, 8))
                recon_1(p, ITERATIONS, recon_type=CT_RECON, noisy=noisy, num_angles=c * katz_len)
    else: 

        for c in [0.5, 1, 2, 4]:

            #set up plots
            err_fig, err_axs = plt.subplots(1, 3, figsize=FIG_SIZE_ERR)
            err_axs[0].tick_params(axis='both', labelsize=20)
            err_axs[1].tick_params(axis='both', labelsize=20)
            err_axs[2].tick_params(axis='both', labelsize=20)
            # err_fig.suptitle(str(c) + " x Katz angles", 
            #                  fontsize=TITLE_FONT_SIZE)

            recon_fig, recon_axs = plt.subplots(3, 2, figsize=FIG_SIZE_RECON)
            recon_axs[2][0].set_xlabel("No noise", fontsize=XY_LABEL_FONT_SIZE)
            recon_axs[2][1].set_xlabel("Noise", fontsize=XY_LABEL_FONT_SIZE)
            recon_axs[0][0].set_ylabel("Regular angle set", fontsize=XY_LABEL_FONT_SIZE)
            recon_axs[1][0].set_ylabel("Prime angle set", fontsize=XY_LABEL_FONT_SIZE)
            recon_axs[2][0].set_ylabel("Composite angle set", fontsize=XY_LABEL_FONT_SIZE)

            diff_fig, diff_axs = plt.subplots(3, 2, figsize=FIG_SIZE_RECON)
            diff_axs[2][0].set_xlabel("No noise", fontsize=XY_LABEL_FONT_SIZE)
            diff_axs[2][1].set_xlabel("Noise", fontsize=XY_LABEL_FONT_SIZE)
            diff_axs[0][0].set_ylabel("Regular angle set", fontsize=XY_LABEL_FONT_SIZE)
            diff_axs[1][0].set_ylabel("Prime angle set", fontsize=XY_LABEL_FONT_SIZE)
            diff_axs[2][0].set_ylabel("Composite angle set", fontsize=XY_LABEL_FONT_SIZE)


            frac_fig, frac_axs = plt.subplots(3, 1, figsize=FIG_SIZE_FRAC)
            lena, mask = imageio.phantom(N, p, True, np.uint32, True)


            for a, noisy in enumerate((False, True)):
                #plot errors
                path = get_path(CT_RECON, 1, c * katz_len, ITERATIONS, noisy=noisy)
                plot_recon_1(path, noisy, err_axs)

                #plot recons
                data = np.load(path)["data"].item()
                im = diff_axs[0][a].imshow(abs(lena-data["regular"]["recon"]), cmap='gray')
                plt.colorbar(im, ax=diff_axs[0, a], cmap='gray')
                im = diff_axs[1][a].imshow(abs(lena-data["prime"]["recon"]), cmap='gray')
                plt.colorbar(im, ax=diff_axs[1, a], cmap='gray')
                im = diff_axs[2][a].imshow(abs(lena-data["composite"]["recon"]), cmap='gray')
                plt.colorbar(im, ax=diff_axs[2, a], cmap='gray')

                im = recon_axs[0][a].imshow(data["regular"]["recon"], cmap='gray',
                                            vmin=0, vmax = 256)
                plt.colorbar(im, ax=recon_axs[0, a], cmap='gray')
                im = recon_axs[1][a].imshow(data["prime"]["recon"], cmap='gray', 
                                            vmin=0, vmax = 256)
                plt.colorbar(im, ax=recon_axs[1, a], cmap='gray')
                im = recon_axs[2][a].imshow(data["composite"]["recon"], cmap='gray', 
                                            vmin=0, vmax = 256)
                plt.colorbar(im, ax=recon_axs[2, a], cmap='gray')

                #plot fractals
                if noisy: #only plot once
                    frac_axs[0].set_xlabel("Regular angle set", fontsize=LABEL_FONT_SIZE)
                    plotFractal(p, data["regular"]["angles"], CT_RECON, frac_axs[0], "Regular subset")
                    frac_axs[1].set_xlabel("Prime angle set", fontsize=LABEL_FONT_SIZE)
                    plotFractal(p, data["prime"]["angles"], CT_RECON, frac_axs[1], "Prime subset")
                    frac_axs[2].set_xlabel("Composite angle set", fontsize=LABEL_FONT_SIZE)
                    plotFractal(p, data["composite"]["angles"], CT_RECON, frac_axs[2], "Composite subset")

                    frac_fig.tight_layout()
                    handles, labels = err_axs[0].get_legend_handles_labels()
                    err_fig.legend(handles, labels, loc='lower center', ncol = 3, 
                                fontsize=LABEL_FONT_SIZE)
                    err_fig.tight_layout(pad = PAD_ERR, rect = RECT_ERR)
                recon_fig.tight_layout()
                diff_fig.tight_layout()
                frac_fig.tight_layout()
                    
                err_fig.savefig('results_CT/recon_2/err_fig_' + (str(c) if c != 0.5 else 'half') + '_.png')
                recon_fig.savefig('results_CT/recon_2/recon_fig_' + (str(c) if c != 0.5 else 'half') + '_.png')
                diff_fig.savefig('results_CT/recon_2/diff_fig_' + (str(c) if c != 0.5 else 'half') + '_.png')
                # frac_fig.savefig('frac_fig_' + (str(c) if c != 0.5 else 'half') + '_.png')


            

            



            
            # cmap = plt.cm.gray
            # norm = plt.Normalize(vmin=0, vmax=50)
            # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            # sm.set_array([])
            # recon_fig.colorbar(sm, ax=recon_axs.ravel().tolist())


def exp_2_1(run):
    """From exp_2, we know that primes are important, but in that scenario, we 
    still have more prime projections than composite - what happens as the 
    number of porjections increase and hence the ratio of composites to prime. 
    We assume that the reconstruction with the most projections should produce 
    the best results, so we excpect the composite reconstruction to out do the 
    prime. 
    Here we test this theory for some multiple of the Kazt criterion. 
    """ 
    if run:
        p = nt.nearestPrime(N)
        for noisy in (True, False): 
            colours = iter(plt.cm.gist_rainbow(np.linspace(0,1,10)))
            for c in [1, 10, 100]:
                angles, _ = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES) 
                katz_num_angles = len(angles)
                plt.figure(figsize=(16, 8))
                recon_1(p, ITERATIONS, recon_type=CT_RECON, noisy=noisy, num_angles=c*katz_num_angles, colours=[next(colours), next(colours), next(colours)])
    else: 
        for noisy in (True, False): 
            angles, _ = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES) 
            katz_num_angles = len(angles)
            for c in [1, 10, 100]:
                path = get_path(CT_RECON, 1, c*katz_num_angles, ITERATIONS, noisy=noisy)
                plot_recon_1(path, noisy)
                plt.suptitle(str(c) + " x Katz angles", fontsize=20)


def exp_3(run):
    """If prime angles are key to reconstruction, what happens when we replace 
    composite numbers with primes which are very close. 

    We get a very similar reconstruction! (is this obvious lol)
    """
    p = nt.nearestPrime(N)

    if run: 
        for noisy in (True, False):
            recon_3(p, KATZ_ANGLES, ITERATIONS, CT_RECON, noisy=noisy)
    else:
        err_fig, err_axs = plt.subplots(1, 3, figsize=FIG_SIZE_ERR)
        err_axs[0].tick_params(axis='both', labelsize=20)
        err_axs[1].tick_params(axis='both', labelsize=20)
        err_axs[2].tick_params(axis='both', labelsize=20)

        recon_fig, recon_axs = plt.subplots(2, 2, figsize=[2*6, 2*6])
        recon_axs[1][0].set_xlabel("No noise", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[1][1].set_xlabel("Noise", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[0][0].set_ylabel("ChaoS angle set", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[1][0].set_ylabel("Composite replacement angle set", fontsize=XY_LABEL_FONT_SIZE)

        diff_fig, diff_axs = plt.subplots(2, 2, figsize=[2*6, 2*6])
        diff_axs[1][0].set_xlabel("No noise", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[1][1].set_xlabel("Noise", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[0][0].set_ylabel("ChaoS angle set", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[1][0].set_ylabel("Composite replacement angle set", fontsize=XY_LABEL_FONT_SIZE)

        lena, mask = imageio.phantom(N, p, True, np.uint32, True)
        # recon_fig.suptitle(str(c) + " x Katz Reconstructions", 
        #                    fontsize=TITLE_FONT_SIZE)

        #set up labels
        

        for a, noisy in enumerate((False, True)):
            path = get_path(CT_RECON, 3, KATZ_ANGLES, ITERATIONS, noisy)
            plot_recon_3(path, noisy, err_axs)

            data = np.load(path)["data"].item()
            reg_recon = data["regular"]
            angle_rep_recon = data["prime_replacement"]

            im = recon_axs[0][a].imshow(reg_recon["recon"], cmap='gray', 
                                            vmin=0, vmax=256)
            plt.colorbar(im, ax=recon_axs[0, a], cmap='gray')
            im = recon_axs[1][a].imshow(angle_rep_recon["recon"], cmap='gray', 
                                            vmin=0, vmax=256)
            plt.colorbar(im, ax=recon_axs[1, a], cmap='gray')

            im = diff_axs[0][a].imshow(abs(lena - reg_recon["recon"]), cmap='gray', 
                                    vmin=0, vmax=15)
            plt.colorbar(im, ax=diff_axs[0, a], cmap='gray')
            im = diff_axs[1][a].imshow(abs(lena - angle_rep_recon["recon"]), cmap='gray', 
                                    vmin=0, vmax=15)
            plt.colorbar(im, ax=diff_axs[1, a], cmap='gray')


        handles, labels = err_axs[0].get_legend_handles_labels()
        err_fig.legend(handles, labels, loc='lower center', ncol = 2, 
                fontsize=LABEL_FONT_SIZE)
        err_fig.tight_layout(pad = PAD_ERR, rect = RECT_ERR)
        recon_fig.tight_layout()
        diff_fig.tight_layout()

        err_fig.savefig('results_CT/recon_3/err_fig_.png')
        recon_fig.savefig('results_CT/recon_3/recon_fig_.png')
        diff_fig.savefig('results_CT/recon_3/diff_fig_.png')

def exp_4(run): 
    """Reconsturction of primes + composite 

    This is all about how much value the composites add. 
    """
    if run: 
        # for noisy in (True, False): 
        noisy = True
        p = nt.nearestPrime(N)
        recon_2(p, KATZ_ANGLES, ITERATIONS, CT_RECON, noisy)
    else: 
        # for noisy in (True, False): 
        noisy = True
        path = get_path(CT_RECON, 2, KATZ_ANGLES, ITERATIONS, noisy)
        plot_recon_2(path, noisy, plot_type=False)


def exp_5(run): 
    """Evidences that the prime reconstruction has a lower Kazt, and so results
    in an equivalently good reconstruction (comparing to regular recon) with 
    less projections, and a better reconstruction that the regular recon with 
    same number of angles (particularly evident with noise).
    """
    p = nt.nearestPrime(N)

    if run: 
        for noisy in (True, False): 
            recon_1c(p, ITERATIONS, recon_type=CT_RECON, noisy=noisy)

    else: 
        #set up plots
        err_fig, err_axs = plt.subplots(1, 3, figsize=FIG_SIZE_ERR)
        title = "Prime angle set comparison" 
        # err_fig.suptitle(title, fontsize = TITLE_FONT_SIZE)

        recon_fig, recon_axs = plt.subplots(3, 2, figsize=FIG_SIZE_RECON)
        # recon_fig.suptitle(title, fontsize=TITLE_FONT_SIZE)
        recon_axs[2][0].set_xlabel("No noise", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[2][1].set_xlabel("Noise", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[0][0].set_ylabel("ChaoS angle set", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[1][0].set_ylabel("Prime angle subset", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[2][0].set_ylabel("Prime angle set", fontsize=XY_LABEL_FONT_SIZE)

        diff_fig, diff_axs = plt.subplots(3, 2, figsize=FIG_SIZE_RECON)
        diff_axs[2][0].set_xlabel("No noise", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[2][1].set_xlabel("Noise", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[0][0].set_ylabel("ChaoS angle set", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[1][0].set_ylabel("Prime angle subset", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[2][0].set_ylabel("Prime angle set", fontsize=XY_LABEL_FONT_SIZE)

        frac_fig, frac_axs = plt.subplots(3, 1, figsize=FIG_SIZE_FRAC)
        lena, mask = imageio.phantom(N, p, True, np.uint32, True)

        for a, noisy in enumerate((False, True)): 
            if noisy:
                path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, noisy, snr = 95)
            else:
                path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, noisy)
            plot_recon_1c(path, noisy, err_axs)

            handles, labels = err_axs[0].get_legend_handles_labels()
            
            #plot recons
            data = np.load(path)["data"].item()
            im = recon_axs[0][a].imshow(data["regular"]["recon"], cmap='gray', 
                                       vmin=0, vmax=256)
            plt.colorbar(im, ax=recon_axs[0, a], cmap='gray')
            im = recon_axs[1][a].imshow(data["prime_subset"]["recon"], cmap='gray', 
                                       vmin=0, vmax=256)
            plt.colorbar(im, ax=recon_axs[1, a], cmap='gray')
            im = recon_axs[2][a].imshow(data["prime_katz"]["recon"], cmap='gray', 
                                       vmin=0, vmax=256)
            plt.colorbar(im, ax=recon_axs[2, a], cmap='gray')

            im = diff_axs[0][a].imshow(abs(lena - data["regular"]["recon"]), cmap='gray', 
                                       vmin=0, vmax=15)
            plt.colorbar(im, ax=diff_axs[0, a], cmap='gray')
            im = diff_axs[1][a].imshow(abs(lena - data["prime_subset"]["recon"]), cmap='gray', 
                                       vmin=0, vmax=15)
            plt.colorbar(im, ax=diff_axs[1, a], cmap='gray')
            im = diff_axs[2][a].imshow(abs(lena - data["prime_katz"]["recon"]), cmap='gray', 
                                       vmin=0, vmax=15)
            plt.colorbar(im, ax=diff_axs[2, a], cmap='gray')

            if noisy: #only plot once
                frac_axs[0].set_xlabel("ChaoS angle set", fontsize=LABEL_FONT_SIZE)
                plotFractal(p, data["regular"]["angles"], CT_RECON, frac_axs[0], "Regular subset")
                frac_axs[1].set_xlabel("Prime subset", fontsize=LABEL_FONT_SIZE)
                plotFractal(p, data["prime_subset"]["angles"], CT_RECON, frac_axs[1], "Prime subset")
                frac_axs[2].set_xlabel("Prime angle set", fontsize=LABEL_FONT_SIZE)
                plotFractal(p, data["prime_katz"]["angles"], CT_RECON, frac_axs[2], "Composite subset")


    err_fig.legend(handles, labels, fontsize = LABEL_FONT_SIZE, 
                        loc='lower center', ncol = 3)
    err_fig.tight_layout(pad = PAD_ERR, rect = RECT_ERR)
    recon_fig.tight_layout()
    diff_fig.tight_layout()
    frac_fig.tight_layout()

    err_fig.savefig('results_CT/recon_5/err_fig_.png')
    recon_fig.savefig('results_CT/recon_5/recon_fig_.png')
    diff_fig.savefig('results_CT/recon_5/diff_fig_.png')
    frac_fig.savefig('results_CT/recon_5/frac_fig_.png')



            

def exp_5b(run): 
    """test out how bas the noise can for prime recon"""
    p = nt.nearestPrime(N)

    if run: 
        # recon_1c(p, ITERATIONS, recon_type=CT_RECON, noisy=False)
        for snr in [0.92, 0.94, 0.96, 0.98]: 
            recon_1c(p, ITERATIONS, recon_type=CT_RECON, noisy=True, snr=snr)
    else:
        #set up plots
        err_fig, err_axs = plt.subplots(1, 3, figsize=FIG_SIZE_ERR)
        err_axs[0].tick_params(axis='both', labelsize=20)
        err_axs[1].tick_params(axis='both', labelsize=20)
        err_axs[2].tick_params(axis='both', labelsize=20)


        title = "Regular (57 projections) and Prime (47 projections) angle set noise response" 
        # err_fig.suptitle(title, fontsize = TITLE_FONT_SIZE)
        
        prime_recon_fig, prime_recon_axs = plt.subplots(2, 3, figsize=FIG_SIZE_RECON[::-1])
        # prime_recon_fig.suptitle("Prime angle set reconstruction", 
        #                          fontsize=TITLE_FONT_SIZE)
        prime_recon_axs = prime_recon_axs.flat


        reg_recon_fig, reg_recon_axs = plt.subplots(2, 3, figsize=FIG_SIZE_RECON[::-1])

        # reg_recon_fig.suptitle("Regular angle set reconstruction", 
        #                          fontsize=TITLE_FONT_SIZE)
        reg_recon_axs = reg_recon_axs.flat

        colour=iter(plt.cm.rainbow(np.linspace(0,1,7)))
        for a, snr in enumerate([1, 0.98, 0.96, 0.94, 0.92, 0.90]):
            noisy = True if snr != 1 else False

            path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, noisy, snr=int(100 * snr))
            data = np.load(path)["data"].item()
            prime_kazt = data["prime_katz"]
            reg = data["regular"]

            c = next(colour)
    
            plot_recon(prime_kazt["rmse"], prime_kazt["psnr"], prime_kazt["ssim"], axs=err_axs,
                    colour = c, 
                    line = "--",
                    label=str("Prime angle set " + str(100 * (1 - snr))) + "% noise")
            
            plot_recon(reg["rmse"], reg["psnr"], reg["ssim"], axs=err_axs,
                    colour = c, 
                    line = "-",
                    label=str("Regular angle set " + str(100 * (1 - snr))) + "% noise")
            
            #plot recons
            lena, mask = imageio.phantom(N, p, True, np.uint32, True)

            im = prime_recon_axs[a].imshow(prime_kazt["recon"], cmap='gray')
            plt.colorbar(im, ax = prime_recon_axs[a], cmap='gray')
            prime_recon_axs[a].set_xlabel(str(100 * (1 - snr)) + "% Noise", fontsize=XY_LABEL_FONT_SIZE)

            im = reg_recon_axs[a].imshow(reg["recon"], cmap='gray')
            plt.colorbar(im, ax = reg_recon_axs[a], cmap='gray')
            reg_recon_axs[a].set_xlabel(str(100 * (1 - snr)) + "% Noise", fontsize=XY_LABEL_FONT_SIZE)
        
        handles, labels = err_axs[0].get_legend_handles_labels()
        err_fig.legend(handles, labels, loc='lower center', ncol = 4, 
                        fontsize=LABEL_FONT_SIZE)
        err_fig.tight_layout(pad = PAD_ERR, rect = RECT_ERR)
        prime_recon_fig.tight_layout()
        reg_recon_fig.tight_layout()


        

def exp_6(run): 
    
    if run: 
        for noisy in (True, False): 
            data = {}
            p = nt.nearestPrime(N)
            angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=KATZ_ANGLES, prime_only=True)
            recon, rmses, psnrs, ssims  = recon_CT(p, angles, subsetAngles, ITERATIONS, noisy)
            data["prime"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

            angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=len(angles), prime_only=False)
            recon, rmses, psnrs, ssims  = recon_CT(p, angles, subsetAngles, ITERATIONS, noisy)
            data["regular"] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

            path = get_path(CT_RECON, 6, len(angles), ITERATIONS, noisy)
            np.savez(file=path, data=data)
    else: 
        for noisy in (True, False): 
            line = '-' if not noisy else "--"
            p = nt.nearestPrime(N)
            angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=KATZ_ANGLES, prime_only=True)
            print(len(angles))
            path = get_path(CT_RECON, 6, len(angles), ITERATIONS, noisy)
            data = np.load(path)["data"].item()
            reg_recon = data["regular"]
            prime_recon = data["prime"]

            plot_recon(reg_recon["rmse"], reg_recon["psnr"], reg_recon["ssim"], 
                    colour = REGULAR, 
                    line = line,
                    label="Regular reconstruction {}".format("(noisy) " if noisy else "") + str(len(reg_recon["angles"])) + " projections", max_it=400)
            
            plot_recon(prime_recon["rmse"], prime_recon["psnr"], prime_recon["ssim"], 
                    colour = PRIME, 
                    line = line,
                    label="Prime reconstruction {}".format("(noisy) " if noisy else "") + str(len(prime_recon["angles"])) + " projections", max_it=400)


def exp_7(run): 
    if run: 
        for noisy in (True, False): 
            p = nt.nearestPrime(N)
            recon_2b(p, KATZ_ANGLES, ITERATIONS, CT_RECON, noisy)
    else: 
        for noisy in (True, False): 
            path = get_path(CT_RECON, "2b", KATZ_ANGLES, ITERATIONS, noisy)
            plot_recon_2(path, noisy, plot_type=False)


def exp_8(run): 
    """recon for more and more primes"""
    if run: 
        data = {}
        noisy = False
        p = nt.nearestPrime(N)
        prime, subset_primes = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=KATZ_ANGLES, prime_only=True)

        #initalise angle sets
        angles = []
        subset_angles = []
        for subset in subset_primes: 
            subset_angles.append([])

        for i in range(len(angles)):
            new_angle = angles[i]
            idx = get_subset_index(new_angle, subset_primes)
            angles.append(new_angle)
            subset_angles[idx].append(new_angle)

            if i > 2: 
                recon, rmses, psnrs, ssims = recon_CT(p, angles, subset_angles, ITERATIONS, noisy=noisy)
                data[i] = {"angles": angles, "recon": recon, "rmse": rmses, "psnr": psnrs, "ssim": ssims, "noise":noisy}

        path = get_path(CT_RECON, 8, len(prime), ITERATIONS, noisy)
        np.savez(file=path, data=data)
    else: 
        p = nt.nearestPrime(N)
        prime, subset_primes = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=KATZ_ANGLES, prime_only=True)
        path = get_path(CT_RECON, 8, len(prime), ITERATIONS, noisy)
        data = np.load(path)['data'].item()

        colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(data.keys())+1)))
        for len_angles, info in data.items(): 
            plot_recon(info["rmse"], info["psnr"], info["ssim"], label="num angles: " + str(len_angles), colour=next(colour))


def poster_plots(plot_recons, plot_errors): 
    p = nt.nearestPrime(N)
    
    if plot_recons: 
        
        lena, mask = imageio.phantom(N, p, True, np.uint32, True)
        fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
        axs = axs.flat
        for ax in axs: 
            ax.axis("off")

        path = get_path(CT_RECON, 1, 1 * 57, ITERATIONS, noisy=False)
        data = np.load(path)["data"].item()
        axs[0].imshow(data["regular"]["recon"], cmap="gray")
        axs[0].set_title("ChaoS recon")
        axs[4].imshow(np.abs(lena - data["regular"]["recon"]), cmap="gray")

        path = get_path(CT_RECON, 1, 1 * 57, ITERATIONS, noisy=True)
        data = np.load(path)["data"].item()
        axs[1].imshow(data["regular"]["recon"], cmap="gray")
        axs[1].set_title("ChaoS recon (noisy)")
        axs[5].imshow(np.abs(lena - data["regular"]["recon"]), cmap="gray")

        path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, False, snr=100)
        data = np.load(path)["data"].item()
        axs[2].imshow(data["prime_katz"]["recon"], cmap="gray")
        axs[2].set_title("ChaoS recon")
        axs[6].imshow(np.abs(lena - data["prime_katz"]["recon"]), cmap="gray")

        path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, True, snr=95)
        data = np.load(path)["data"].item()
        axs[3].imshow(data["prime_katz"]["recon"], cmap="gray")
        axs[3].set_title("ChaoS recon (noisy)")
        axs[7].imshow(np.abs(lena - data["prime_katz"]["recon"]), cmap="gray")

    if plot_errors: 
        cm = 1/2.54 * 3
        plt.figure(figsize=(15*cm, 5.5*cm))
        colours = iter(plt.cm.rainbow(np.linspace(0,1,6)))

        path = get_path(CT_RECON, 1, 1 * 57, ITERATIONS, noisy=True)
        data = np.load(path)["data"].item()
        plot_recon_1(path, True, [next(colours), next(colours), next(colours)])

        path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, True, snr=95)
        plot_recon_1c(path, True, [next(colours), next(colours)])

        colours = iter(plt.cm.rainbow(np.linspace(0,1,6)))

        path = get_path(CT_RECON, 1, 1 * 57, ITERATIONS, noisy=False)
        data = np.load(path)["data"].item()
        plot_recon_1(path, False, [next(colours), next(colours), next(colours)])

        path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, False)
        plot_recon_1c(path, False, [next(colours), next(colours)])


def vis_projs(): 
    lena, mask = imageio.phantom(N, p, True, np.uint32, True)

    #get half plane of angles
    angles, subsetAngles = extend_quadrant(subsetAngles)

    mt_lena = mojette.transform(lena, angles)
    
    #add noise 
    if noisy:
        add_noise(mt_lena, SNR_CT) 

    rt_lena = mojette.toDRT(mt_lena, angles, p, p, p) 


def angle_fractal_contributions(): 
    p = nt.nearestPrime(N)
    angles, subset_angles = angleSubSets_Symmetric(s,subsetsMode,p,p,K=K, max_angles=-1, prime_only=True)
    fig, axs = plt.subplots(5, len(angles)//5 + 1)
    axs = axs.flat

    for j, angle in enumerate(angles): 
        lines, mValues = calcFiniteLines([angle], p)
        powerSpect = np.zeros((p,p))
        maxLines = len(lines)
        #maxLines = 12
        fareyImage = np.zeros_like(powerSpect)
        for i, line in enumerate(lines):
            u, v = line
            fareyImage[u,v] = 255
            if i == maxLines:
                break
        axs[j].imshow(fareyImage)
        axs[j].set_title(str(angle))


def mri_ct(run): 
    p = nt.nearestPrime(N)
    angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES)  

    if run:
        data = {}
        recon__CT,  rmse, psnrs, ssims = recon_CT(p, angles, subsetAngles, ITERATIONS)
        data["CT"] = {"recon":recon__CT,  "RMSE":rmse, "PSNR":psnrs, "SSIM":ssims, "angles":angles, "noisy":False}
        recon__MRI,  rmse, psnrs, ssims = recon_MRI(p, angles, subsetAngles, ITERATIONS)
        data["MRI"] = {"recon":recon__MRI,  "RMSE":rmse, "PSNR":psnrs, "SSIM":ssims, "angles":angles, "noisy":False}
        path = get_path(CT_RECON, 0, len(angles), ITERATIONS, noisy=False)
        np.savez(file=path, data=data)

        data = {}
        err_fig, err_axs = plt.subplots(1, 3)
        recon_CT_noisy,  rmse, psnrs, ssims = recon_CT(p, angles, subsetAngles, ITERATIONS, True, snr=0.95)
        data["CT"] = {"recon":recon_CT_noisy,  "RMSE":rmse, "PSNR":psnrs, "SSIM":ssims, "angles":angles, "noisy":True, "SNR":0.95}
        recon_MRI_noisy,  rmse, psnrs, ssims = recon_MRI(p, angles, subsetAngles, ITERATIONS, True)
        data["MRI"] = {"recon":recon_MRI_noisy,  "RMSE":rmse, "PSNR":psnrs, "SSIM":ssims, "angles":angles, "noisy":True, "SNR":0.95}
        path = get_path(CT_RECON, 0, len(angles), ITERATIONS, noisy=True)
        np.savez(file=path, data=data)

    else:
        err_fig, err_axs = plt.subplots(1, 3, figsize=FIG_SIZE_ERR)
        err_axs[0].tick_params(axis='both', labelsize=20)
        err_axs[1].tick_params(axis='both', labelsize=20)
        err_axs[2].tick_params(axis='both', labelsize=20)
        # title = "Choatic sensing reconstruction for MRI and CT" 
        # err_fig.suptitle(title, fontsize = TITLE_FONT_SIZE)

        recon_fig, recon_axs = plt.subplots(2, 2, figsize=FIG_SIZE_RECON_2_2)
        recon_axs[0][0].set_ylabel("No noise", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[1][0].set_ylabel("Noise", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[1][0].set_xlabel("CT", fontsize=XY_LABEL_FONT_SIZE)
        recon_axs[1][1].set_xlabel("MRI", fontsize=XY_LABEL_FONT_SIZE)

        diff_fig, diff_axs = plt.subplots(2, 2, figsize=FIG_SIZE_RECON_2_2)
        diff_axs[0][0].set_ylabel("No noise", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[1][0].set_ylabel("Noise", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[1][0].set_xlabel("CT", fontsize=XY_LABEL_FONT_SIZE)
        diff_axs[1][1].set_xlabel("MRI", fontsize=XY_LABEL_FONT_SIZE)

        lena, mask = imageio.phantom(N, p, True, np.uint32, True)
        # recon_fig.suptitle(title, fontsize=TITLE_FONT_SIZE)

        for a, noisy in enumerate([False, True]):

            path = get_path(CT_RECON, 0, len(angles), ITERATIONS, noisy=noisy)
            data = np.load(path)["data"].item()

            line = "--" if noisy else "-"

            #plot errors
            plot_recon(data["CT"]["RMSE"], data["CT"]["PSNR"], data["CT"]["SSIM"], 
                    err_axs, colour="hotpink", 
                    line = line,
                    label="CT" + (" (noise)" if noisy else ""))
            plot_recon(data["MRI"]["RMSE"], data["MRI"]["PSNR"], data["MRI"]["SSIM"], 
                    err_axs, colour="skyblue", 
                    line = line,
                    label="MRI" + (" (noise)" if noisy else ""))
            
            #plot errors
            im = recon_axs[a][0].imshow(data["CT"]["recon"], cmap='gray', vmin=0, vmax=256)
            plt.colorbar(im, ax=recon_axs[0, a], cmap='gray')
            im = recon_axs[a][1].imshow(abs(data["MRI"]["recon"]), cmap='gray', vmin=0, vmax=256)
            plt.colorbar(im, ax=recon_axs[1, a], cmap='gray')

            #plot errors
            im = diff_axs[a][0].imshow(abs(lena - data["CT"]["recon"]), cmap='gray', 
                                       vmin = 0, vmax = 15)
            plt.colorbar(im, ax=diff_axs[0, a], cmap='gray')
            im = diff_axs[a][1].imshow(abs(lena - abs(data["MRI"]["recon"])), cmap='gray', 
                                       vmin = 0, vmax = 15)
            plt.colorbar(im, ax=diff_axs[1, a], cmap='gray')
        
        handles, labels = err_axs[0].get_legend_handles_labels()
        err_fig.legend(handles, labels, fontsize = LABEL_FONT_SIZE, 
                           loc='lower center', ncol = 4)
        err_fig.tight_layout(pad = PAD_ERR, rect = RECT_ERR)
        recon_fig.tight_layout()
        diff_fig.tight_layout()

        err_fig.savefig('results_CT/recon_0/err_fig_.png')
        recon_fig.savefig('results_CT/recon_0/recon_fig_.png')
        diff_fig.savefig('results_CT/recon_0/diff_fig_.png')


def final_prime_set(run): 
    exp_5(run)


def demo(run): 
    """Evidences that the prime reconstruction has a lower Kazt, and so results
    in an equivalently good reconstruction (comparing to regular recon) with 
    less projections, and a better reconstruction that the regular recon with 
    same number of angles (particularly evident with noise).
    """
    noisy = True
    if run: 
        p = nt.nearestPrime(N)
        recon_1c(p, ITERATIONS, recon_type=CT_RECON, noisy=noisy)

    else: 
        path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, noisy)
        plot_recon_1c(path, noisy)


def final_comp():
    #set up plots
    err_fig, err_axs = plt.subplots(1, 3, figsize = FIG_SIZE_ERR)
    err_axs[0].tick_params(axis='both', labelsize=20)
    err_axs[1].tick_params(axis='both', labelsize=20)
    err_axs[2].tick_params(axis='both', labelsize=20)
    title = "Prime angle set comparison" 
    # err_fig.suptitle(title, fontsize = TITLE_FONT_SIZE)

    lena, mask = imageio.phantom(N, p, True, np.uint32, True)
    image = imread(sk.data_dir + "/phantom.png", as_grey=True)

    for a, noisy in enumerate((False, True)): 
        recon_fig, recon_axs = plt.subplots(2, 2, figsize = FIG_SIZE_RECON_2_2)
        recon_axs = recon_axs.flat

        diff_fig, diff_axs = plt.subplots(2, 2, figsize = FIG_SIZE_RECON_2_2)
        diff_axs = diff_axs.flat

        if noisy:
            path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, noisy, snr = 95)
        else:
            path = get_path(CT_RECON, "1c", KATZ_ANGLES, ITERATIONS, noisy)
        plot_recon_1c(path, noisy, err_axs)

        handles, labels = err_axs[0].get_legend_handles_labels()
        
        data = np.load(path)["data"].item()

        FBP_recon = 256 * (data["FBP"]["recon"] / np.max(data["FBP"]["recon"]))
        SART_recon = 256 * (data["FBPSART"]["recon"] / np.max(data["FBPSART"]["recon"]))

        im = recon_axs[0].imshow(data["regular"]["recon"], cmap='gray', 
                                 vmin=0, vmax=256)
        recon_axs[0].set_xlabel("ChaoS", fontsize=LABEL_FONT_SIZE)
        plt.colorbar(im, ax=recon_axs[0], cmap='gray')
        im = recon_axs[1].imshow(data["prime_katz"]["recon"], cmap='gray', 
                                 vmin=0, vmax=256)
        recon_axs[1].set_xlabel("Prime ChaoS", fontsize=LABEL_FONT_SIZE)
        plt.colorbar(im, ax=recon_axs[1], cmap='gray')
        im = recon_axs[2].imshow(FBP_recon, cmap='gray', 
                                 vmin=0, vmax=256)
        recon_axs[2].set_xlabel("FBP", fontsize=LABEL_FONT_SIZE)
        plt.colorbar(im, ax=recon_axs[2], cmap='gray')
        im = recon_axs[3].imshow(SART_recon, cmap='gray', 
                                 vmin=0, vmax=256)
        recon_axs[3].set_xlabel("SART", fontsize=LABEL_FONT_SIZE)
        plt.colorbar(im, ax=recon_axs[3], cmap='gray')

        im = diff_axs[0].imshow(abs(lena - data["regular"]["recon"]), cmap='gray')
        plt.colorbar(im, ax=diff_axs[0], cmap='gray')
        diff_axs[0].set_xlabel("ChaoS", fontsize=LABEL_FONT_SIZE)
        im = diff_axs[1].imshow(abs(lena - data["prime_katz"]["recon"]), cmap='gray')
        plt.colorbar(im, ax=diff_axs[1], cmap='gray')
        diff_axs[1].set_xlabel("Prime ChaoS", fontsize=LABEL_FONT_SIZE)
        im = diff_axs[2].imshow(abs(lena - FBP_recon), cmap='gray')
        plt.colorbar(im, ax=diff_axs[2], cmap='gray')
        diff_axs[2].set_xlabel("FBP", fontsize=LABEL_FONT_SIZE)
        im = diff_axs[3].imshow(abs(lena - SART_recon), cmap='gray')
        plt.colorbar(im, ax=diff_axs[3], cmap='gray')
        diff_axs[3].set_xlabel("SART", fontsize=LABEL_FONT_SIZE)

        recon_fig.tight_layout()
        diff_fig.tight_layout()
        recon_fig.savefig('results_CT/recon_fig_' + ("noise" if noisy else "") + '.png')
        diff_fig.savefig('results_CT/diff_fig_' + ("noise" if noisy else "") + '.png')

    
    err_fig.legend(handles, labels, fontsize = LABEL_FONT_SIZE, 
                        loc='lower center', ncol = 4)
    err_fig.tight_layout(pad = PAD_ERR, rect = RECT_ERR)
    err_fig.savefig('results_CT/err_fig_.png')
        



#Shes a runner shes a track star -----------------------------------------------
RUN = True
PLOT = False

TITLE_FONT_SIZE = 30
XY_LABEL_FONT_SIZE = 20
LABEL_FONT_SIZE = 19
FIG_SIZE_ERR = [3*6.9, 3*3.9]
FIG_SIZE_RECON = [3*6, 3*3.9][::-1]
FIG_SIZE_RECON_2_2 = [2*6, 2*6]
FIG_SIZE_FRAC = [3*3.9, 3*11]
PAD_ERR = 0.2
RECT_ERR = (0.057, 0.13, 0.986, 0.974)
# RECT_RECON = ()

UQ_PURPLE = "#51247a"
UQ_GRAY = "#999490"
UQ_RED = "#e62645"
UQ_BLUE = "#4085c6"
UQ_YELLOW = "#fbb800"


if __name__ == "__main__": 
    p = nt.nearestPrime(N)

    



    # for n in [2, 16, 32, 64, 128, 256, 512]:
    #     recon, _, _, _ = fbp(p, n)
    #     axs = plt.imshow(recon, cmap="gray")
    #     plt.gca().set_axis_off()
    #     path = "fbp_" + str(n) + ".png"
    #     plt.savefig(path)
    #     plt.figure()

    # mri_ct(RUN)
    # mri_ct(PLOT)

    # final_comp()

    #run comp + prime + reg - done
    # exp_2(RUN)
    # exp_2(PLOT)

    #run prime katz angle set
    # exp_5(RUN)
    # exp_5(PLOT)

    #run prime katz with changing noise
    # exp_5b(RUN)
    # exp_5b(PLOT)

    #run prime replacement - done
    # exp_3(RUN)
    # exp_3(PLOT)

    #run min comp set - need reg?
    # exp_4(RUN)
    # exp_4(PLOT)

    # exp_7(PLOT)

    # exp_2_1(PLOT)

    # exp_6(RUN)
    # exp_6(PLOT)
    # angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=KATZ_ANGLES) 
    # angles_, recon, rmses, psnrs, ssims = regular_recon(p, angles, subsetAngles, iterations, CT_RECON, noisy=True, colour="orange")

    # angles, subsetAngles = angleSubSets_Symmetric(s,subsetsMode,N,N,K=K, max_angles=47)  
    # angles_, recon, rmses, psnrs, ssims = regular_recon(p, angles, subsetAngles, iterations, CT_RECON, noisy=True, colour="hotpink")


    plt.show()