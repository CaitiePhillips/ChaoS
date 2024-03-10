# %%
from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import numpy as np
from matplotlib import pyplot as plt 
import finitetransform.imageio as imageio #local module
import scipy.fftpack as fftpack
import finitetransform.numbertheory as nt #local modules
import finitetransform.radon as radon
from scipy import ndimage
import finitetransform.farey as farey #local module
import finitetransform.mojette as mojette
import pyfftw
pyfftw.interfaces.cache.enable()
# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

N = 256
k = 1
M = k * N
prime_p = nt.nearestPrime(M)

# %%
def check_drt(mri_path, ct_path): 
    data = np.load(mri_path)
    mri_angles = data['angles']
    mri_drt = np.abs(data['drt'])

    data = np.load(ct_path)
    ct_angles = data['angles']
    ct_drt = data['drt']
    plt.imshow([mri_drt[-1], mri_drt[-1], mri_drt[-1], ct_drt[-1],ct_drt[-1], ct_drt[-1]]) # why is the last one so bad lol


    # fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

    # ax[0].imshow(mri_drt)
    # ax[1].imshow(ct_drt)
    # ax[2].imshow(abs(mri_drt[:-1] - ct_drt[:-1])) # why is the last one so bad lol
    plt.show()

def ct_given_mri(mri_path): 
    lena, mask = imageio.phantom(N, prime_p, True, np.uint32, True)
    data = np.load(mri_path)

    mri_angles = data['angles']
    ct_angles = []
    for angle in mri_angles: 
        ct_angles.append(angle)
        p, q = farey.get_pq(angle)
        if farey.farey(p, - 1 * q) not in ct_angles:
            ct_angles.append(farey.farey(p, - 1 * q))
    print('angles:', ct_angles)

    mValues = []
    for angle in ct_angles:
        m, inv = farey.toFinite_1(angle, prime_p)
        mValues.append(m)
    print("mVals\n", mValues)

    mt_lena = mojette.transform(lena, ct_angles)
    ct_drt = mojette.toDRT(mt_lena, ct_angles, prime_p, N, N) # was p, N, N
    mri_drt = np.abs(data['drt'])

    dbl_drt = []
    for m in mValues:
        print(m)
        dbl_drt.append(mri_drt[m])
        dbl_drt.append(mri_drt[m])
        dbl_drt.append(ct_drt[m])
        dbl_drt.append(ct_drt[m])
        dbl_drt.append(np.zeros_like(ct_drt[m]))

    plt.imshow(dbl_drt)
    plt.show()


if __name__ == "__main__":
    ct_given_mri("MRI_results.npz")