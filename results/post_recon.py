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
import time
import math

import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt 
from scipy import ndimage


def plot_errors(angles, error):#, psnr_all, ssim_all):
    # for prime in range(2):
    #     for i, rmse in enumerate(rmse_all[prime]): 
    #         plt.plot(rmse, label="angles: " + str(test_angles[i]) + " prime: " + str(bool(prime)))

    for i in range(len(error[0])): 
        plt.figure()
        for prime in range(2):
            label = "prime" if prime == 0 else "composite"
            plt.plot(np.sqrt(error[prime][i]), label=label)
        plt.legend()
        plt.title("RMSE, angles = " + str(angles[i]))
        plt.savefig("results/auto_recon_2/PSNR_100its_" + str(angles[i]) + "angles.png")
            

data = np.load("results/auto_recon_2/recon_its_100.npz")
angles = [15, 25, 50, 75, 100]
psnr = data["psnr"]
plot_errors(angles, psnr)
print("done")