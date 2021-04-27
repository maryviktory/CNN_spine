import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt

CUTOFF = [0.1, 2]  # expresed in Hz
FS = 1000  # sampling freqeuncy


def filter(y):
    # First, design the Buterworth filter
    N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, y)
    return filt

def open_file(name_file):
    train_file = open(name_file, "r")
    lines_train = train_file.read().splitlines()
    print('patient names:',lines_train)
    return lines_train

# patients = open_file('/home/maria/IFL_project/forceultrasuondspine/Spinous_counting/dataset_to_plot.txt')

# patients = "PolyU_sweep"
# for patient_name in patients:

    # patient_name = 'Maria_V'
frame = pd.read_csv("/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/Subjects dataset/sweep012_Spinous_Probability_label.csv")
x = np.array(frame.loc[:, "Spinous Probability"])
y = np.array(frame.loc[:, "Label"])
# z = np.array(frame.loc[:, "Force Z (N)"])

# x_filt = filter(x)
x_filt = x



# mod = np.sqrt(np.square(x) + np.square(y) + np.square(z))
# mod_filt = np.sqrt(np.square(x_filt) + np.square(y_filt) + np.square(z_filt))

# plt.figure()
# ax1 = plt.subplot(2, 2, 1)
# ax1.set_title("X axis filt")
# ax1.plot(x_filt)
#
# ax2 = plt.subplot(2, 2, 2)
# ax2.set_title("Y axis filt")
# ax2.plot(y_filt)

# ax3 = plt.subplot(2, 2, 3)
# ax3.set_title("Z axis filt")
# ax3.plot(z_filt)

# ax4 = plt.subplot(2, 2, 4)
# ax4.set_title("modulus filt")
# ax4.plot(mod_filt)

plt.figure()
ax1 = plt.subplot(2, 1, 1)
# ax1.set_title("CNN probabilities")
plt.xlabel('Frames',fontsize=15)
plt.ylabel("CNN probabilities",fontsize=15)
ax1.plot(1 - x_filt)

ax2 = plt.subplot(2, 1, 2)
# ax2.set_title("Labels")
plt.ylabel("Label",fontsize=15)
plt.xlabel('Frames',fontsize=15)
ax2.plot(y)

# ax1 = plt.subplot(2, 1, 1)
# # ax1.set_title("CNN probabilities")
# plt.xlabel('Frames',fontsize=15)
# plt.ylabel("CNN probabilities",fontsize=15)
# plt.plot(x_filt, label = 'CNN probability')
# plt.plot(y, 'r', label = 'Label')
# plt.legend(fontsize=15)



# ax3 = plt.subplot(2, 2, 3)
# ax3.set_title("Z axis")
# ax3.plot(z)

# ax4 = plt.subplot(2, 2, 4)
# ax4.set_title("modulus")
# ax4.plot(mod)

plt.show()