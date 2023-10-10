# -*- coding: utf-8 -*-
"""
Finite measurement module for MRI data

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
import _libpath #add custom libs
import finitetransform.radon as radon
import finitetransform.farey as farey #local module
import scipy.fftpack as fftpack
import pyfftw
import numpy as np
import math

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def computeLines(kSpace, angles, centered = True, twoQuads = False):
    '''
    compute finite lines coordinates
    Returns a list or list of slice 2-tuples and corresponding list of m values
    '''
    p, s = kSpace.shape
    lines = []
    mValues = []
    for angle in angles:
        m, inv = farey.toFinite(angle, p)
        u, v = radon.getSliceCoordinates2(m, kSpace, centered, p)
        lines.append((u,v))
        mValues.append(m)
        #second quadrant
        if twoQuads:
            if m != 0 and m != p: #dont repeat these
                m = p-m
                u, v = radon.getSliceCoordinates2(m, kSpace, centered, p)
                lines.append((u,v))
                mValues.append(m)
    
    return lines, mValues

def frt(image, N, dtype=np.float32, mValues=None):
    '''
    Compute the DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be an NxN image to recover the DRT projections (bins), where N is prime.
    Float type is returned by default to ensure no round off issues.
    '''
    mu = N+1
        
    #FFT image
    fftLena = fftpack.fft2(image) #the '2' is important
    
    bins = np.zeros((mu,N),dtype=dtype)
    for m in range(0, mu):
        if mValues and m not in mValues:
            continue 
        
        slice = radon.getSlice(m, fftLena)
#        print slice
#        slice /= N #norm FFT
        projection = np.real(fftpack.ifft(slice))
        #Copy and norm
        for j in range(0, N):
            bins[m, j] = projection[j]
    
    return bins
    
def frt_complex(image, N, dtype=np.complex, mValues=None, center=False):
    '''
    Compute the DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be an NxN image to recover the DRT projections (bins), where N is prime.
    Float type is returned by default to ensure no round off issues.
    '''
    mu = N+1
        
    #FFT image
    fftLena = fftpack.fft2(image) #the '2' is important
    
    bins = np.zeros((mu,N),dtype=dtype)
    for m in range(0, mu):
        if mValues and m not in mValues:
            continue 
        
        slice = radon.getSlice(m, fftLena)
#        print slice
#        slice /= N #norm FFT
        projection = fftpack.ifft(slice)
        if center:
            projection = fftpack.ifftshift(projection)
        #Copy and norm
        for j in range(0, N):
            bins[m, j] = projection[j]
    
    return bins

def ifrt(bins, N, norm = True, center = False, projNumber = 0, Isum = -1, mValues=None):
    '''
    Compute the inverse DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be DRT projections (bins) to recover an NxN image, where N is prime.
    projNumber is the number of non-zero projections in bins. This useful for backprojecting mu projections where mu < N.
    Isum is computed from first row if -1, otherwise provided value is used
    '''
    if Isum < 0:
        Isum = bins[0,:].sum()
#    print "ISUM:", Isum
    dftSpace = np.zeros((N,N),dtype=np.complex)
    
    #Set slices (0 <= m <= N)
    for k, row in enumerate(bins): #iterate per row
        if mValues and k not in mValues:
            continue
        
        slice = fftpack.fft(row)
        radon.setSlice(k,dftSpace,slice)
#    print "filter:", filter
    dftSpace[0,0] -= float(Isum)*N

    #iFFT 2D image
    result = fftpack.ifft2(dftSpace)
    if not norm:
        result *= N #ifft2 already divides by N**2
    if center:
        result = fftpack.fftshift(result)

    return np.real(result)
    
def ifrt_complex(bins, N, norm = True, center = False, projNumber = 0, Isum = -1, mValues=None):
    '''
    Compute the inverse DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be DRT projections (bins) to recover an NxN image, where N is prime.
    projNumber is the number of non-zero projections in bins. This useful for backprojecting mu projections where mu < N.
    Isum is computed from first row if -1, otherwise provided value is used
    '''
    if Isum < 0:
        Isum = bins[0,:].sum()
#    print "ISUM:", Isum
    dftSpace = np.zeros((N,N),dtype=np.complex)
    
    #Set slices (0 <= m <= N)
    for k, row in enumerate(bins): #iterate per row
        if mValues and k not in mValues:
            continue
        
        slice = fftpack.fft(row)
        radon.setSlice(k,dftSpace,slice)
#    print "filter:", filter
    dftSpace[0,0] -= float(Isum)*N

    #iFFT 2D image
    result = fftpack.ifft2(dftSpace)
    if not norm:
        result *= N #ifft2 already divides by N**2
    if center:
        result = fftpack.ifftshift(result)

    return result

def mse(img1, img2):
    '''
    Compute the MSE of two images using mask if given
    '''
    error = ((img1 - img2) ** 2).mean(axis=None)
    
    return error
    
def psnr(img1, img2, maxPixel=255):
    '''
    Compute the MSE of two images using mask if given
    '''
    error = mse(img1,img2)
    psnr_out = 20 * math.log(maxPixel / math.sqrt(error), 10)
    
    return psnr_out

#import random

def noise(kSpace, snr):
    '''
    Create noise in db for given kSpace and SNR
    '''
    r, s = kSpace.shape
    #pwoer of signal and noise
    P = np.sum(np.abs(kSpace)**2)/(r*s)
    P_N = P / (10**(snr/10))
    #P_N is equivalent to sigma**2 and signal usually within 3*sigma
    sigma = math.sqrt(P_N)
    
    noise = np.zeros_like(kSpace)
    for u, row in enumerate(kSpace):
        for v, coeff in enumerate(row):
            noiseReal = np.random.normal(0, sigma)
            noiseImag = np.random.normal(0, sigma)
            noise[u,v] = noiseReal + 1j*noiseImag
            
    return noise

def noise_mt(proj, snr):
    '''
    Create noise in db for given kSpace and SNR
    '''
    r = proj.size
    #pwoer of signal and noise
    # P = np.sum(np.abs(proj)**2)/(r) # this overflows
    P = np.sum((np.abs(proj)**2) / r)
    P_N = P / (10**(snr/10))
    #P_N is equivalent to sigma**2 and signal usually within 3*sigma
    sigma = math.sqrt(P_N)
    
    noise = np.zeros_like(proj)
    for u, val in enumerate(proj):
            noiseReal = np.abs(np.random.normal(0, sigma)) # no -ve projections
            # noiseImag = np.random.normal(0, sigma)
            # for MT, we are working w real numbers so should have any im component
            noise[u] = noiseReal
            
    return noise
