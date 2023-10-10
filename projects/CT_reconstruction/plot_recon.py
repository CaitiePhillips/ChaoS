
import numpy as np
from matplotlib import pyplot as plt

iterations_lst = [500, 1000, 1500]
addNoise_lst = [True, False]
angles_lst = [113, 209, 337, 449]

fig, ax_true = plt.subplots(nrows=3, ncols=4)
fig_false, ax_false = plt.subplots(nrows=3, ncols=4)

for r, iterations in enumerate(iterations_lst): 
    for c, max_angles in enumerate(angles_lst):
            file_true = "noise_True_its_{}_angles_{}.npz".format(iterations, max_angles)
            file_false = "noise_False_its_{}_angles_{}.npz".format(iterations, max_angles)
            data_true = np.load(file_true)
            data_false = np.load(file_false)
            recon_true = np.array(data_true["recon"])
            recon_false = np.array(data_false["recon"])
            time_true = data_true["time"]
            time_false = data_false["time"]

            ax_true[r, c].imshow(recon_true)
            ax_true[r, c].set_title("recon time: {:.5}s".format(time_true))
            ax_false[r, c].imshow(recon_false)
            ax_false[r, c].set_title("recon time: {:.5}s".format(time_false))

plt.show()