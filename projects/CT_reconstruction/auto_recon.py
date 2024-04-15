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
SNR = 30
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
    ax_rmse.set_title("RMSE")
    ax_ssims.set_title("SSIM")
    ax_psnr.set_title("PSNR")

    labels = ["regular (gaussian integer) recon", "gaussian prime recon", "gaussian integer w/o prime recnon"]
    paths = ["CT_results/exp_0/regular_recon_angles_20_its_1000.npz", 
            "CT_results/exp_0/prime_recon_angles_20_its_1000.npz", 
            "CT_results/exp_0/composite_recon_angles_20_its_1000.npz"]
    for i, path in enumerate(paths): 
        data = np.load(path)
        ax_rmse.plot(data["rmses"])
        ax_ssims.plot(data["ssims"])
        ax_psnr.plot(data["psnrs"], label=labels[i])
        ax_psnr.legend()
    plt.show()
    

def plot_recons(recon_info, reg_recon=None, prime_recon=None): 
    fig, (ax_rmse, ax_ssims, ax_psnr) = plt.subplots(1, 3)
    fig.set_size_inches(14, 7)
    fig.tight_layout()
    if reg_recon: 
        ax_rmse.plot(reg_recon["rmses"])
        ax_ssims.plot(reg_recon["ssims"])
        ax_psnr.plot(reg_recon["psnrs"], label="regular recon")
    if prime_recon: 
        ax_rmse.plot(prime_recon["rmses"])
        ax_ssims.plot(prime_recon["ssims"])
        ax_psnr.plot(prime_recon["psnrs"], label="prime recon")

    for label, error_info in recon_info.items(): 
        ax_rmse.plot(error_info["rmses"])
        ax_ssims.plot(error_info["ssims"])
        ax_psnr.plot(error_info["psnrs"], label=label)
        ax_psnr.legend()
        

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
        if type == 1: 
            subsets.append([farey.farey(p, q), farey.farey(p, -1*q)]) #type 1 - mirror vertical
        elif type == 2: 
            subsets.append([farey.farey(q, p), farey.farey(q, -1*p)]) #type 2 - + 90deg
        # elif type == 3: 
        #     subsets.append([farey.farey(p, q), farey.farey(q, p)]) #type 3 - mirror diag
        elif type == 4: 
            subsets.append([farey.farey(p, q), farey.farey(q, p), 
                            farey.farey(q, -1*p), farey.farey(p, -1*q)]) #type 4 - all quads 
    return subsets


def auto_recon_1(p, num_angles, iterations, type=4): 
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
    QUADS = 2
    angles, subsetAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles)    
    primes, subsetPrimes, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles, prime_only=True)    
    composites, subsetComposites = get_composites(subsetAngles)
    compositesGrouped = get_compostie_sets(composites, type)

    
    for i, compositeSet in enumerate(compositesGrouped): 
        #add angles to prime subset
        subsetAngles = list(subsetPrimes)
        angles = list(primes)
        for angle in compositeSet:
            idx = get_subset_index(angle, subsetComposites)
            subsetAngles[idx].append(angle)
            angles.append(angle)
        #recon with new angle set & save
        recon, rmses, psnrs, ssims = recon_CT(p, angles, subsetAngles, iterations)
        to_plot[str(compositeSet)] = {"composites":compositeSet, "recon":recon, "rmses":rmses, "psnrs":psnrs, "ssims":ssims}
        


    path = "CT_results/exp_1/prime_comp_recons_type_" + str(type) + "_angles_"+ str(num_angles) + "_its_" + str(iterations) + "_copy.npz"
    np.savez(path, primes=primes, plotInfo=to_plot)
    plot_recons(to_plot)
   

def auto_recon_1b(p, num_angles, iterations, type=4):
    """
    complete reconstruction for 
        angles = prime angles (from the reg angles) + (a, b) + (-a, b) (almost same as auto_recon_1)
    for each set of (a, b)s in the composite list. 


    Args:
        p (int): size of reconstruction and fractal
        num_angles (int): number of angles to use in reconstruction
        iterations (int): number of osem iterations
    """
    to_plot = {}

    angles, subsetAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,QUADS,True,K, max_angles=num_angles)    
    primes, subsetPrimes = get_primes(subsetAngles)
    composites, subsetComposites = get_composites(subsetAngles)
    compositesGrouped = get_compostie_sets(composites, type)

    
    for i, compositeSet in enumerate(compositesGrouped): 
        #add angles to prime subset
        subsetAngles = list(subsetPrimes)
        angles = list(primes)
        for angle in compositeSet:
            idx = get_subset_index(angle, subsetComposites)
            subsetAngles[idx].append(angle)
            angles.append(angle)
        #recon with new angle set & save
        recon, rmses, psnrs, ssims = recon_CT(p, angles, subsetAngles, iterations)
        to_plot[str(compositeSet)] = {"composites":compositeSet, "recon":recon, "rmses":rmses, "psnrs":psnrs, "ssims":ssims}
        print(str(i) + "/" + str(len(composites) / type))


    path = "CT_results/exp_1/prime_comp_recons_type_" + str(type) + "_angles_"+ str(num_angles) + "_its_" + str(iterations) + "_copy.npz"
    np.savez(path, primes=primes, plotInfo=to_plot)
    plot_recons(to_plot)


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


def plot_recon_1_each_angle(): 
    """Plot the reconstructions for each quad/pair of composite angles additional
    to the prime angle set. 
    """
    reg_recon = np.load("CT_results/exp_0/regular_recon_angles_300_its_300.npz")

    prime_recon = np.load("CT_results/exp_0/prime_recon_angles_300_its_300.npz")

    data = np.load("CT_results/exp_1/prime_comp_recons_type_1_angles_20_its_300.npz")
    type_1_data = data["plotInfo"].item()

    data = np.load("CT_results/exp_1/prime_comp_recons_type_2_angles_20_its_300.npz")
    type_2_data = data["plotInfo"].item()

    data = np.load("CT_results/exp_1/prime_comp_recons_type_3_angles_20_its_300.npz")
    type_3_data = data["plotInfo"].item()

    data = np.load("CT_results/exp_1/prime_comp_recons_type_4_angles_20_its_300.npz")
    type_4_data = data["plotInfo"].item()

    axs = {}
    set_plots = True
    for data in [type_1_data, type_2_data, type_3_data, type_4_data]:
        
        # colour = iter(plt.cm.jet(np.linspace(0,1,9)))

        for label, recon_info in data.items(): 
            p, q = farey.get_pq(recon_info["composites"][0])
            # due to the angle types, there will be two angles to look out for 
            base_angle = farey.farey(abs(p), abs(q))
            secondary_angle = farey.farey(abs(q), abs(p)) 

            # create plot / get plot axis for angle
            if base_angle not in axs and secondary_angle in axs:
                base_angle = secondary_angle
            elif set_plots:
                fig, ax = plt.subplots(2, 2)
                (ax_rmse, ax_ssim, ax_psnr, ax_angle) = ax.flatten()
                axs[base_angle] = {"figure":fig, "ax_rmse":ax_rmse, "ax_ssim":ax_ssim, "ax_psnr":ax_psnr, "ax_angle":ax_angle}

                #include baseline reconstructions
                ax_rmse.plot(reg_recon["rmses"], label="regular recon")
                ax_ssim.plot(reg_recon["ssims"], label="regular recon")
                ax_psnr.plot(reg_recon["psnrs"], label="regular recon")

                ax_rmse.plot(prime_recon["rmses"], label="prime recon")
                ax_ssim.plot(prime_recon["ssims"], label="prime recon")
                ax_psnr.plot(prime_recon["psnrs"], label="prime recon")
                plot_angles(prime_recon["angles"], label="prime", ax=ax_angle)


                # fig.suptitle(str(base_angle))

            #plot
            # c = next(colour)
            axs[base_angle]["ax_rmse"].plot(recon_info["rmses"], label=label)
            axs[base_angle]["ax_ssim"].plot(recon_info["ssims"], label=label)
            axs[base_angle]["ax_psnr"].plot(recon_info["psnrs"], label=label)
            plot_angles(recon_info["composites"], label=label, ax=axs[base_angle]["ax_angle"], colour='r', linewidth=2)
            axs[base_angle]["ax_psnr"].legend()
            
        
        set_plots = False

    for angle, data in axs.items(): 
        plt.figure(data["figure"].number)
        data["figure"].set_size_inches(14, 7)
        data["figure"].tight_layout()
        path = "CT_results/exp_1/indv_angle_plot_w_angles/" + str(angle) + ".png"
        plt.savefig(path, bbox='tight')

    plt.show()


def plot_recon_1_each_type():
    reg_recon = np.load("CT_results/exp_0/regular_recon_angles_300_its_300.npz")
    prime_recon = np.load("CT_results/exp_0/prime_recon_angles_300_its_300.npz")

    for i in range(4): 
        path = "CT_results/exp_1/prime_comp_recons_type_" + str(i + 1) + "_angles_20_its_300.npz"
        data = np.load(path)
        data = data["plotInfo"].item()
        plot_recons(data, reg_recon=reg_recon, prime_recon=prime_recon)

        # plt.show()
        
        # path = "CT_results/exp_1/indv_type_plots/type_" + str(i+1) + ".png"
        # plt.savefig(path)


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


def noise_work(): 
    
    angles, subsetAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,p,p,2,True,K, max_angles=20)    
    noisy = recon_CT(p, angles, subsetAngles, 100, addNoise = True)
    not_noisy = recon_CT(p, angles, subsetAngles, 100, addNoise = False)

    plt.subplot(1, 3, 1)
    plt.imshow(noisy)
    plt.subplot(1, 3, 2)
    plt.imshow(not_noisy)
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(not_noisy - noisy))
    plt.show()
    pass



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


def recon_CT(p, angles, subsetAngles, iterations, addNoise=False): 
    print(addNoise)
    lena, mask = imageio.phantom(N, p, True, np.uint32, True)

    #convert angles to gradients for OSEM
    subsetsMValues = []
    for subset in subsetAngles:
        mValues = []
        for angle in subset:
            m, inv = farey.toFinite(angle, p)
            mValues.append(m)            
        subsetsMValues.append(mValues)
    # print(subsetsMValues)

    mt_lena = mojette.transform(lena, angles)
    
    #add noise 
    if addNoise:
        mt_noise(mt_lena, 40) 

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


def recon_MRI(p, angles, subsetAngles, iterations, addNoise=False): 
    image, mask = imageio.phantom(N, p, True, np.uint32, True)

    #k-space
    fftImage = fftpack.fft2(image) #the '2' is important
    fftImageShifted = fftpack.fftshift(fftImage)

    #power spectrum
    powSpectImage = np.abs(fftImageShifted)

    #add noise to kSpace
    noise = finite.noise(fftImageShifted, SNR)
    if addNoise:
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
    print(subsetsMValues)

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


def plot_recon(rmseValues, psnrValues, ssimValues, colour = "b", line = '-', label="label"):
    incX = np.arange(0, len(rmseValues))*plotIncrement

    plt.subplot(1, 3, 1)
    plt.plot(incX, rmseValues, c=colour, ls=line, label=label)
    plt.title('Error Convergence of the Finite OSEM')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')

    plt.subplot(1, 3, 2)
    plt.plot(incX, psnrValues, c=colour, ls=line, label=label)
    plt.ylim(0, 45.0)
    plt.title('PSNR Convergence of the Finite OSSEM')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')

    plt.subplot(1, 3, 3)
    plt.plot(incX, ssimValues, c=colour, ls=line, label=label)
    plt.ylim(0, 1.0)
    plt.title('Simarlity Convergence of the Finite OSSEM')
    plt.xlabel('Iterations')
    plt.ylabel('SSIM')
    plt.legend()



#helpers -----------------------------------------------------------------------
def remove_empty(subset_angles): 
    return [subset for subset in subset_angles if subset != []]


def mt_noise(mt_projs, snr):
    '''
    Create noise in db for given kSpace and SNR+
    '''
    P_S = 0
    for i, proj in enumerate(mt_projs):
        P_S += np.sum((np.abs(proj)**2) / (len(proj) * len(mt_projs)))

    P_N = P_S / (10**(snr/10))
    sigma = math.sqrt(P_N)
    
    for u, proj in enumerate(mt_projs):
        for v, coeff in enumerate(proj):
            mt_projs[u][v] += np.random.normal(0, sigma)


def rt_noise(rt_projs, snr):
    '''
    Create noise in db for given kSpace and SNR+
    '''
    r, s = rt_projs.shape
    #pwoer of signal and noise
    P = np.sum(np.abs(rt_projs)**2)/(r*s)
    P_N = P / (10**(snr/10))
    #P_N is equivalent to sigma**2 and signal usually within 3*sigma
    sigma = math.sqrt(P_N)
    
    noise = np.zeros_like(rt_projs)
    for u, row in enumerate(rt_projs):
        for v, coeff in enumerate(row):
            noise[u,v] = np.random.normal(0, sigma)
            
    return noise


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


# reconstructions --------------------------------------------------------------
def recon_0(p, num_angles_octant, iterations):
    """Reconstructs and plort MRI and CT reconstruction from same angle set. 
    Prime example of the algorithms applicability to CT reconstruction. 
    """
    num_angles_mri = num_angles_octant * OCTANT_MRI - 1 #must be odd to account for (1, 1) having no mirror pair  
    num_angles_ct = 2 * (num_angles_mri - 2) #each angle maps to itself and its vertical mirror EXCEPT (0, 1) and (1, 0) 

    angles_CT, subsetAngles_CT = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=num_angles_ct) 
    angles_MRI, subsetAngles_MRI = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_MRI,K=K, max_angles=num_angles_mri)  
    # plot_angles(angles_CT, colour="hotpink") 
    # plot_angles(angles_MRI, colour="skyblue", line="-.") 
    # plt.figure()

    recon, mses, psnrs, ssims = recon_CT(p, angles_CT, remove_empty(subsetAngles_CT), iterations)
    plot_recon(mses, psnrs, ssims, colour="hotpink", line=LINE_CT)

    recon, mses, psnrs, ssims = recon_MRI(p, angles_MRI, remove_empty(subsetAngles_MRI), iterations)
    plot_recon(mses, psnrs, ssims, colour="skyblue", line=LINE_MRI)
    # plt.show()


def recon_1(p, num_angles_octant, iterations): 
    """plots reconstructions of regular, prime, and composite angle sets for both CT and MRI

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
    """

    colour=iter(plt.cm.jet(np.linspace(0,1,6+1)))
    data_CT = {}
    data_MRI = {}

    num_angles_mri = num_angles_octant * OCTANT_MRI - 1 #must be odd to account for (1, 1) having no mirror pair  
    num_angles_ct = 2 * (num_angles_mri - 2) #each angle maps to itself and its vertical mirror EXCEPT (0, 1) and (1, 0) 

    #regular angle set
    angles_CT, subset_angles_CT = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=num_angles_ct) 
    angles_MRI, subset_angles_MRI = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_MRI,K=K, max_angles=num_angles_mri)  

    _, rmses_CT, psnrs_CT, ssims_CT = recon_CT(p, angles_CT, subset_angles_CT, iterations)
    _, rmses_MRI, psnrs_MRI, ssims_MRI = recon_MRI(p, angles_MRI, subset_angles_MRI, iterations)

    plot_recon(rmses_CT, psnrs_CT, ssims_CT, colour=next(colour), label="regular CT")
    plot_recon(rmses_MRI, psnrs_MRI, ssims_MRI, colour=next(colour), label="regular MRI", line=LINE_MRI)
    
    #save
    data_CT["regular"] = {"angles": angles_CT, "rmse": rmses_CT, "psnr": psnrs_CT, "ssim": ssims_CT}
    data_MRI["regular"] = {"angles": angles_MRI, "rmse": rmses_MRI, "psnr": psnrs_MRI, "ssim": ssims_MRI}


    #primes in each angle set 
    primes_CT, primes_subset_CT = get_primes(subset_angles_CT)
    primes_MRI, primes_subset_MRI = get_primes(subset_angles_MRI)

    _, rmses_CT, psnrs_CT, ssims_CT = recon_CT(p, primes_CT, remove_empty(primes_subset_CT), iterations)
    _, rmses_MRI, psnrs_MRI, ssims_MRI = recon_MRI(p, primes_MRI, remove_empty(primes_subset_MRI), iterations)

    plot_recon(rmses_CT, psnrs_CT, ssims_CT, colour=next(colour), label="prime CT")
    plot_recon(rmses_MRI, psnrs_MRI, ssims_MRI, colour=next(colour), label="prime MRI", line=LINE_MRI)

    #save
    data_CT["prime"] = {"angles": primes_CT, "rmse": rmses_CT, "psnr": psnrs_CT, "ssim": ssims_CT}
    data_MRI["prime"] = {"angles": primes_MRI, "rmse": rmses_MRI, "psnr": psnrs_MRI, "ssim": ssims_MRI}

    #composites in each angle set 
    comps_CT, comps_subset_CT = get_composites(subset_angles_CT)
    comps_MRI, comps_subset_MRI = get_composites(subset_angles_MRI)

    _, rmses_CT, psnrs_CT, ssims_CT = recon_CT(p, comps_CT, remove_empty(comps_subset_CT), iterations)
    _, rmses_MRI, psnrs_MRI, ssims_MRI = recon_MRI(p, comps_MRI, remove_empty(comps_subset_MRI), iterations)

    plot_recon(rmses_CT, psnrs_CT, ssims_CT, colour=next(colour), label="composite CT")
    plot_recon(rmses_MRI, psnrs_MRI, ssims_MRI, colour=next(colour), label="composite MRI", line=LINE_MRI)

    #save
    data_CT["composite"] = {"angles": comps_CT, "rmse": rmses_CT, "psnr": psnrs_CT, "ssim": ssims_CT}
    data_MRI["composite"] = {"angles": comps_MRI, "rmse": rmses_MRI, "psnr": psnrs_MRI, "ssim": ssims_MRI}

    path = "results_CT/recon_1/oct_angles_" + str(num_angles_octant) + "_its_" + str(iterations) + ".npz"
    np.savez(file=path, data_CT=data_CT)
    path = "results_MRI/recon_1/oct_angles_" + str(num_angles_octant) + "_its_" + str(iterations) + ".npz"
    np.savez(file=path, data_CT=data_MRI)

    plt.show()


def recon_2(p, num_angles_octant, iterations, type = 4):
    """Recconstructs with prime angle set + pair/quad set of composite angles. 
    Plots and saves. 

    Args:
        p (int): prime size of image
        num_angles_octant (int): number of angles per octant
        iterations (int): number of OSEM iterations
        type (int, optional): Octants to identify equivalent angle in, 1, 2, or 4. Defaults to 4.
    """
    data_CT = {}
    data_MRI = {}

    #angles for top two quadrants
    num_angles_MRI = 2 * num_angles_octant - 1  
    num_angles_CT = 2 * (num_angles_MRI - 2)
    angles_CT, subset_angles_CT = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=num_angles_CT) 
    angles_MRI, subset_angles_MRI = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_MRI,K=K, max_angles=num_angles_MRI)  

    #composites in top two quadrants 
    composites, subset_composites = get_composites(subset_angles_CT)
    octant_composites = get_compostie_sets(composites, type) # sort into octant equivalence classes (same angle in different octants)
    colour=iter(plt.cm.gist_rainbow(np.linspace(0,1,len(2 * octant_composites)+3)))

    recon, rmses_CT, psnrs_CT, ssims_CT = recon_CT(p, angles_CT, remove_empty(subset_angles_CT), iterations)
    plot_recon(rmses_CT, psnrs_CT, ssims_CT, label="regular CT", colour=next(colour))

    recon, rmses_MRI, psnrs_MRI, ssims_MRI = recon_MRI(p, angles_MRI, remove_empty(subset_angles_MRI), iterations)
    plot_recon(rmses_MRI, psnrs_MRI, ssims_MRI, label="regular MRI", colour=next(colour))

    for i, comp_equiv_class in enumerate(octant_composites):
        #reset compositie + prime subset
        new_angles_CT, new_subset_angles_CT = get_primes(subset_angles_CT) #this is stupid it won't seperate the lists even if i list(prime_subset_CT) >:(
        new_angles_MRI, new_subset_angles_MRI = get_primes(subset_angles_MRI)
        
        #build compositie + prime subset
        for octant in range(type):
            #add at same index as in regular subset
            angle = comp_equiv_class[octant]
            idx = get_subset_index(angle, subset_composites)
            new_subset_angles_CT[idx].append(angle)
            new_angles_CT.append(angle)

            #only adding first quadrant angles to MRI subset
            if octant < OCTANT_MRI: 
                angles_MRI.append(angle)
                new_subset_angles_MRI[idx].append(angle)
                new_angles_MRI.append(angle)

        #reconstruct with new compositie + prime subset 
        recon, rmses_CT, psnrs_CT, ssims_CT = recon_CT(p, new_angles_CT, remove_empty(new_subset_angles_CT), iterations)
        plot_recon(rmses_CT, psnrs_CT, ssims_CT, label=str(comp_equiv_class)+" CT", colour=next(colour))
        data_CT[str(comp_equiv_class)] = {"angles": new_angles_CT, "rmse": rmses_CT, "psnr": psnrs_CT, "ssim": ssims_CT}

        recon, rmses_MRI, psnrs_MRI, ssims_MRI = recon_MRI(p, new_angles_MRI, remove_empty(new_subset_angles_MRI), iterations)
        plot_recon(rmses_MRI, psnrs_MRI, ssims_MRI, label=str(comp_equiv_class)+" MRI", colour=next(colour))
        data_MRI[str(comp_equiv_class)] = {"angles": new_angles_MRI, "rmse": rmses_MRI, "psnr": psnrs_MRI, "ssim": ssims_MRI}
        

    path = "results_CT/recon_2/oct_angles_" + str(num_angles_octant) + "_its_" + str(iterations) + "_type_" + str(type) + ".npz"
    np.savez(file=path, data_CT=data_CT)
    path = "results_MRI/recon_2/oct_angles_" + str(num_angles_octant) + "_its_" + str(iterations) + "_type_" + str(type) + ".npz"
    np.savez(file=path, data_CT=data_MRI)

    plt.show()


def recon_3(p, num_angles_octant, iterations)
    pass


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




#Shes a runner shes a track star -----------------------------------------------
#recon constants
NUM_OCTANT_ANGLES = 20
ITERATIONS = 100
OCTANT_MRI = 2
OCTANT_CT = 4
#plotting constants
LINE_MRI = "-"
LINE_CT = '-'



if __name__ == "__main__": 
    p = nt.nearestPrime(N)
    recon_4(p, NUM_OCTANT_ANGLES, ITERATIONS)

    # num_angles_mri = NUM_OCTANT_ANGLES * OCTANT_MRI - 1 #must be odd to account for (1, 1) having no mirror pair  
    # num_angles_ct = 2 * (num_angles_mri - 2) #each angle maps to itself and its vertical mirror EXCEPT (0, 1) and (1, 0) 

    # angles_CT, subset_angles_CT = angleSubSets_Symmetric(s,subsetsMode,p,p,octant=OCTANT_CT,K=K, max_angles=num_angles_ct) 
    
    # lena, mask = imageio.phantom(N, p, True, np.uint32, True)

    # mt_lena = mojette.transform(lena, angles_CT)
    # rt_lena = mojette.toDRT(mt_lena, angles_CT, p, p, p) 
    
    # plt.imshow(rt_noise(rt_lena, 40))
    # plt.show()