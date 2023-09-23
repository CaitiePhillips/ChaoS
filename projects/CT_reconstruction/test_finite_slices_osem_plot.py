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
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means
import scipy.fftpack as fftpack
import pyfftw
import numpy as np
import finite
import time
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import matplotlib
matplotlib.use('Qt4Agg')

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()
# Define the colors for the colormap
colors = [(0, 0, 0), (1, 0, 0)]  # Black and Pink
custom_cmap = LinearSegmentedColormap.from_list("black_red", colors, N=256)

#parameter sets (K, k, i, s, h)
#phantom
#parameters = [1.2, 1, 381, 30, 8.0] #r=2
parameters = [0.4, 1, 760, 12, 12.0] #r=4
#cameraman
#parameters = [1.2, 1, 381, 30, 8.0] #r=2

#parameters
n = 257 #prime for not dyadic size
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
angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,N,N,1,True,K)
#angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,M,M,1,True,K)
perpAngle = farey.farey(1,0)
angles.append(perpAngle)
subsetsAngles[0].append(perpAngle)
print("Number of Angles:", len(angles))
print("angles:", angles)

p = nt.nearestPrime(M)
print("p:", p)

#check if Katz compliant
if not mojette.isKatzCriterion(N, N, angles):
    print("Warning: Katz Criterion not met")

#create test image
#lena, mask = imageio.lena(N, p, True, np.uint32, True)
lena, mask = imageio.phantom(N, p, True, np.uint32, True)
#lena, mask = imageio.cameraman(N, p, True, np.uint32, True)

#-------------------------------
# # k-space
# #2D FFT
# print("Creating kSpace")
# fftLena = fftpack.fft2(lena) #the '2' is important
# fftLenaShifted = fftpack.fftshift(fftLena)
# #power spectrum
# powSpectLena = np.abs(fftLenaShifted)
# %%
print("Creating sinogram")
mt_lena = mojette.transform(lena, angles)
rt_lena = mojette.toDRT(mt_lena, angles, N, N, N) #sinogram
# plt.imshow(rt_lena)
# plt.axis("off")
# plt.show()

#experimenting with mojette, plot projections
# pad each mt so same length to plot
max_len = max([len(mt) for mt in mt_lena])
mt_padded = np.zeros((N + 1, max_len))
for i, mt in enumerate(mt_lena): 
    m, inv = farey.toFinite(angles[i], N)

    mt = np.array(mt_lena[i])
    mt_padded[m] = np.pad(mt, (max_len - len(mt))//2, 'constant')

# plt.imshow(mt_padded)
# plt.axis("off")
# plt.show()

# print("angles:", angles)
# im = mojette.backproject(mt_lena, angles, N, N)

# v

# %%
#compute slopes of given angles
powSpectLena = np.zeros((N, N)) # empty 2DFT
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
print("Number of lines:", mu)
print(subsetsMValues)

# fill in the 2D FT
mValues = [m for mSet in subsetsMValues for m in mSet] #FIXME: this is not simple complexity wise i think
for m in mValues: 
    slice = fftpack.fft(rt_lena[m])
    radon.setSlice(m, powSpectLena, slice)

powSpectLena = fftpack.fftshift(powSpectLena)
powSpectLena = powSpectLena + np.flipud(powSpectLena) # conj symmetric
result = fftpack.ifft2(powSpectLena) 
# lena_fractal = max(20 * np.log10(abs(powSpectLena)), 20)
lena_fractal = [[min(abs(i), 1) for i in array ] for array in powSpectLena]

figure, axis = plt.subplots(1, 3)

axis[0].imshow(rt_lena)
axis[0].set_title("DRT")
axis[0].axis("off")

axis[1].imshow(lena_fractal, cmap = custom_cmap)
axis[1].set_title("2D FT")
axis[1].axis("off")

axis[2].imshow(np.abs(result), cmap='gray')
axis[2].set_title("Recon")
axis[2].axis("off")
plt.show()
# #-------------
# # Measure finite slice
# from scipy import ndimage

# print("Measuring slices")
# drtSpace = np.zeros((p+1, p), floatType)
# for lines, mValues in zip(subsetsLines, subsetsMValues):
#     for i, line in enumerate(lines):
#         u, v = line
#         sliceReal = ndimage.map_coordinates(np.real(fftLenaShifted), [u,v])
#         sliceImag = ndimage.map_coordinates(np.imag(fftLenaShifted), [u,v])
#         slice = sliceReal+1j*sliceImag
#     #    print("slice", i, ":", slice)
#         finiteProjection = fftpack.ifft(slice) # recover projection using slice theorem
#         drtSpace[mValues[i],:] = finiteProjection
# #print("drtSpace:", drtSpace)

# #-------------------------------
# #define MLEM
# def osem_expand_complex(iterations, p, g_j, os_mValues, projector, backprojector, image, mask, epsilon=1e3, dtype=np.int32):
#     '''
#     # Gary's implementation
#     # From Lalush and Wernick;
#     # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
#     # where g = \sum (h f^\hat)                                   ... (**)
#     #
#     # self.f is the current estimate f^\hat
#     # The following g from (**) is equivalent to g = \sum (h f^\hat)
#     '''
#     norm = False
#     center = False
#     fdtype = floatType
#     f = np.ones((p,p), fdtype)
    
#     mses = []
#     psnrs = []
#     ssims = []
#     for i in range(0, iterations):
#         print("Iteration:", i)
#         for j, mValues in enumerate(os_mValues):
# #            print("Subset:", j)
#             muFinite = len(mValues)
            
#             g = projector(f, p, fdtype, mValues)
        
#             # form parenthesised term (g_j / g) from (*)
#             r = np.copy(g_j)
#             for m in mValues:
# #                r[m,:] = g_j[m,:] / g[m,:]
#                 for y in range(p):
#                     r[m,y] /= g[m,y]
        
#             # backproject to form \sum h * (g_j / g)
#             g_r = backprojector(r, p, norm, center, 1, 0, mValues) / muFinite
        
#             # Renormalise backprojected term / \sum h)
#             # Normalise the individual pixels in the reconstruction
#             f *= g_r
        
#         if smoothReconMode > 0 and i % smoothIncrement == 0 and i > 0: #smooth to stem growth of noise
#             if smoothReconMode == 1:
#                 print("Smooth TV")
#                 f = denoise_tv_chambolle(f, 0.02, multichannel=False)
#             elif smoothReconMode == 2:
#                 h = parameters[4]
#                 if i > smoothMaxIteration:
#                     h /= 2.0
#                 if i > smoothMaxIteration2:
#                     h /= 4.0
#                 print("Smooth NL h:",h)
#                 fReal = denoise_nl_means(np.real(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
#                 fImag = denoise_nl_means(np.imag(f), patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
#                 f = fReal +1j*fImag
#             elif smoothReconMode == 3:
#                 print("Smooth Median")
#                 fReal = ndimage.median_filter(np.real(f), 3)
#                 fImag = ndimage.median_filter(np.imag(f), 3)
#             f = fReal +1j*fImag
            
#         if i%plotIncrement == 0:
#             img = imageio.immask(image, mask, N, N)
#             recon = imageio.immask(f, mask, N, N)
#             recon = np.abs(recon)
#             mse = imageio.immse(img, recon)
#             psnr = imageio.impsnr(img, recon)
#             ssim = imageio.imssim(img.astype(float), recon.astype(float))
#             print("RMSE:", math.sqrt(mse), "PSNR:", psnr, "SSIM:", ssim)
#             mses.append(mse)
#             psnrs.append(psnr)
#             ssims.append(ssim)
        
#     return f, mses, psnrs, ssims

# #-------------------------------
# #reconstruct test using MLEM   
# start = time.time() #time generation
# recon, mses, psnrs, ssims = osem_expand_complex(iterations, p, drtSpace, subsetsMValues, finite.frt_complex, finite.ifrt_complex, lena, mask)
# recon = np.abs(recon)
# print("Done")
# end = time.time()
# elapsed = end - start
# print("OSEM Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# mse = imageio.immse(imageio.immask(lena, mask, N, N), imageio.immask(recon, mask, N, N))
# ssim = imageio.imssim(imageio.immask(lena, mask, N, N).astype(float), imageio.immask(recon, mask, N, N).astype(float))
# psnr = imageio.impsnr(imageio.immask(lena, mask, N, N), imageio.immask(recon, mask, N, N))
# print("RMSE:", math.sqrt(mse))
# print("SSIM:", ssim)
# print("PSNR:", psnr)
# diff = lena - recon

# #save mat file of result
# #np.savez('result_osem.npz', recon=recon, diff=diff, psnrs=psnrs, ssims=ssims)
# np.savez('result_phantom_osem.npz', recon=recon, diff=diff, psnrs=psnrs, ssims=ssims)
# #np.savez('result_camera_osem.npz', recon=recon, diff=diff, psnrs=psnrs, ssims=ssims)

# #plot
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# #pp = PdfPages('finite_osem_plots.pdf')
# pp = PdfPages('finite_osem_phantom_plots.pdf')
# #pp = PdfPages('finite_osem_camera_plots.pdf')

# fig, ax = plt.subplots(figsize=(24, 9))

# if plotCroppedImages:
#     print(lena.shape)
#     print(mask.shape)
#     lena = imageio.immask(lena, mask, N, N)
#     reconLena = imageio.immask(reconLena, mask, N, N)
#     reconNoise = imageio.immask(reconNoise, mask, N, N)
#     recon = imageio.immask(recon, mask, N, N)
#     diff = imageio.immask(diff, mask, N, N)

# plt.subplot(121)
# rax = plt.imshow(reconLena, interpolation='nearest', cmap='gray')
# #rax = plt.imshow(reconLena, cmap='gray')
# rcbar = plt.colorbar(rax, cmap='gray')
# plt.title('Image (w/ Noise)')
# plt.subplot(122)
# rax2 = plt.imshow(recon, interpolation='nearest', cmap='gray')
# #rax2 = plt.imshow(recon, cmap='gray')
# rcbar2 = plt.colorbar(rax2, cmap='gray')
# plt.title('Reconstruction')
# pp.savefig()

# fig, ax = plt.subplots(figsize=(24, 9))


# plt.subplot(151)
# rax = plt.imshow(lena, interpolation='nearest', cmap='gray')
# #rax = plt.imshow(lena, cmap='gray')
# rcbar = plt.colorbar(rax, cmap='gray')
# plt.title('Image')
# plt.subplot(152)
# rax = plt.imshow(reconLena, interpolation='nearest', cmap='gray')
# #rax = plt.imshow(reconLena, cmap='gray')
# rcbar = plt.colorbar(rax, cmap='gray')
# plt.title('Image (w/ Noise)')
# plt.subplot(153)
# rax = plt.imshow(reconNoise, interpolation='nearest', cmap='gray')
# #rax = plt.imshow(reconNoise, cmap='gray')
# rcbar = plt.colorbar(rax, cmap='gray')
# plt.title('Noise')
# plt.subplot(154)
# rax2 = plt.imshow(recon, interpolation='nearest', cmap='gray')
# #rax2 = plt.imshow(recon, cmap='gray')
# rcbar2 = plt.colorbar(rax2, cmap='gray')
# plt.title('Reconstruction')
# plt.subplot(155)
# rax3 = plt.imshow(diff, interpolation='nearest', cmap='gray')
# #rax3 = plt.imshow(diff, cmap='gray')
# rcbar3 = plt.colorbar(rax3, cmap='gray')
# plt.title('Reconstruction Errors')
# pp.savefig()

# #plot convergence
# fig, ax = plt.subplots(figsize=(24, 9))

# mseValues = np.array(mses)
# psnrValues = np.array(psnrs)
# ssimValues = np.array(ssims)
# incX = np.arange(0, len(mses))*plotIncrement

# plt.subplot(131)
# plt.plot(incX, np.sqrt(mseValues))
# plt.title('Error Convergence of the Finite OSEM')
# plt.xlabel('Iterations')
# plt.ylabel('RMSE')
# plt.subplot(132)
# plt.plot(incX, psnrValues)
# plt.ylim(0, 45.0)
# plt.title('PSNR Convergence of the Finite OSSEM')
# plt.xlabel('Iterations')
# plt.ylabel('PSNR')
# plt.subplot(133)
# plt.plot(incX, ssimValues)
# plt.ylim(0, 1.0)
# plt.title('Simarlity Convergence of the Finite OSSEM')
# plt.xlabel('Iterations')
# plt.ylabel('SSIM')
# pp.savefig()
# pp.close()

# plt.show()

# print("Complete")

# %%
