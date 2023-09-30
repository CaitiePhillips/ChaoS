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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time
import matplotlib
matplotlib.use('Qt4Agg')

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()


# %%
def compute_slopes(subsetsAngles, twoQuads = True): 
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

def fill_dft(subsetsMValues, powSpectLena): 
    mValues = [m for mSet in subsetsMValues for m in mSet] #FIXME: this is not simple complexity wise i think
    for m in mValues: 
        slice = fftpack.fft(rt_lena[m])
        radon.setSlice(m, powSpectLena, slice)

    powSpectLena = fftpack.fftshift(powSpectLena)
    powSpectLena = powSpectLena + np.flipud(powSpectLena) # conj symmetric
    return powSpectLena

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
    smoothReconMode = 2 #0-None,1-TV,2-NL,3-Median
    smoothIncrement = 10
    smoothMaxIteration = iterations/2
    relaxIterationFactor = int(0.01*iterations)
    smoothMaxIteration2 = iterations-relaxIterationFactor*smoothIncrement
    plotIncrement = 2

    norm = False
    center = False
    fdtype = np.complex
    f = np.ones((p,p), fdtype)
    
    for i in range(0, iterations):
        print("Iteration:", i)
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
                fImag = denoise_nl_means(np.imag(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                f = fReal +1j*fImag
            elif smoothReconMode == 3:
                print("Smooth Median")
                fReal = ndimage.median_filter(np.real(f), 3)
                fImag = ndimage.median_filter(np.imag(f), 3)
            f = fReal +1j*fImag
            
        if i%plotIncrement == 0:
            img = imageio.immask(image, mask, N, N)
            recon = imageio.immask(f, mask, N, N)
            recon = np.abs(recon)
        
    return f

data_dft = {}
data_recon = {}
data_radon = {}

iterations = [5, 10]

for j in range(0, 2): 
    #parameter sets (K, k, i, s, h)
    #phantom
    parameters = [1.2, 1, iterations[j], 30, 8.0, j] #r=2
    #Phantom: N=256, i=381, s=30, h=8, K=1.2, k=1;

    #parameters
    n = 257 #prime for not dyadic size
    k = parameters[1]
    M = int(k*n)
    N = n 
    K = parameters[0]
    s = parameters[3]
    subsetsMode = 1

    angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,N,N,1,True,K)
    perpAngle = farey.farey(1,0)
    angles.append(perpAngle)
    subsetsAngles[0].append(perpAngle)
    p = nt.nearestPrime(M)

    #create test image
    lena, mask = imageio.phantom(N, p, True, np.uint32, True)

    #check if Katz compliant
    if not mojette.isKatzCriterion(N, N, angles):
        print("Warning: Katz Criterion not met")
    
    #MT and RT
    mt_lena = mojette.transform(lena, angles)
    rt_lena = mojette.toDRT(mt_lena, angles, N, N, N) #sinogram

    #RT to DFT
    subsetsMValues = compute_slopes(subsetsAngles)
    powSpectLena = np.zeros((N, N)) # empty 2DFT
    powSpectLena = fill_dft(subsetsMValues, powSpectLena)
    lena_fractal = [[min(abs(i), 1) for i in array ] for array in powSpectLena]

    start = time.time() #time generation
    recon = osem_expand_complex(parameters[2], p, rt_lena, subsetsMValues, finite.frt_complex, finite.ifrt_complex, lena, mask)
    recon = np.abs(recon)
    print("Done")
    end = time.time()
    elapsed = end - start
    print("OSEM Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    parameters.append(elapsed)

    data_dft[str(parameters)] = powSpectLena.ravel()
    data_recon[str(parameters)] = recon.ravel()
    
df_dft = pd.DataFrame(data_dft)
df_recon = pd.DataFrame(data_recon)
print("appending")
df_dft.to_csv('dft.csv', index = False)
df_recon.to_csv('recon.csv', index = False)
print("done")
 # %%
colors = [(0, 0, 0), (1, 0, 0)]  # Black and Pink
custom_cmap = LinearSegmentedColormap.from_list("black_red", colors, N=256)

df = pd.read_csv('dft.csv')
figure, axis = plt.subplots(1, 2)
for v in range(0, 2): 
    data_dft = np.array(df.iloc[:, v].tolist())
    data_dft = np.reshape(data_dft, (N, N))
    lena_fractal = [[min(abs(i), 1) for i in array ] for array in data_dft]
    axis[v].imshow(lena_fractal, cmap = custom_cmap)
    axis[v].set_title(df.iloc[:, v].name)
    axis[v].axis("off")

df = pd.read_csv('recon.csv')
figure2, axis2 = plt.subplots(1, 2)
for v in range(0, 2): 
    recon_data = np.array(df.iloc[:, v].tolist())
    recon_data = np.reshape(recon_data, (N, N))
    axis2[v].imshow(recon_data, cmap = 'gray')
    axis2[v].set_title(df.iloc[:, v].name)
    axis2[v].axis("off")
plt.show()

# inverse DFT
# result = fftpack.ifft2(powSpectLena) 

# #plot
# figure, axis = plt.subplots(1, 3)

# axis[0].imshow(rt_lena)
# axis[0].set_title("DRT")
# axis[0].axis("off")


# axis[1].imshow(lena_fractal, cmap = custom_cmap)
# axis[1].set_title("2D FT")
# axis[1].axis("off")

# axis[2].imshow(np.abs(result), cmap='gray')
# axis[2].set_title("Recon")
# axis[2].axis("off")

# plt.show()

# #experimenting with mojette, plot projections
# # pad each mt so same length to plot
# max_len = max([len(mt) for mt in mt_lena])
# mt_padded = np.zeros((N + 1, max_len))
# for i, mt in enumerate(mt_lena): 
#     m, inv = farey.toFinite(angles[i], N)

#     mt = np.array(mt_lena[i])
#     mt_padded[m] = np.pad(mt, (max_len - len(mt))//2, 'constant')

# %%
