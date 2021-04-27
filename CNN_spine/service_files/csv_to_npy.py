import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime
from scipy.interpolate import interp1d
import pandas as pd


sacrum_flag = True
showFlag = True
save = False

if sacrum_flag:
    csv_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/Subjects dataset/data_train/withSacrum/csv"
    save_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/Subjects dataset/data_train/withSacrum/npy"
    train_list = ["sweep012","sweep018","sweep014","sweep015","sweep017","sweep019","sweep3006"]

else:
    csv_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/Subjects dataset/csv"
    save_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/Subjects dataset/npy"
    train_list = ["sweep012", "sweep018", "sweep013", "sweep014", "sweep015", "sweep017", "sweep019", "sweep020",
                  "sweep3005", "sweep3006"]

#CV 1

# Hannes, Hendrik
val_list = []

# Arian, MariaT, Magda, Ardit
test_list = []

not_used = []

spacing = 0.1


def smooth(signal, N=30):
    N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, signal)
    return filt


def undrift(signal):
    N = 2  # Filter order
    Wn = 0.01  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    avg = filtfilt(B, A, signal)
    undrifted = signal - avg

    return undrifted


def get_step_labels(binary_label_list):

    last_label = 0
    vertebral_level = 0
    label_list = []
    for label in binary_label_list:
        if label == 1 and last_label == 0:
            vertebral_level += 1

        label_list.append(vertebral_level)
        last_label = label

    return np.array(label_list)

def main():

    # Getting the list of data
    file_list = [file for file in os.listdir(csv_dir) if ".csv" in file]

    for file in file_list:
        data_array = np.zeros(())
        print("processing: {}".format(file))
        # if "sweep18" not in file:
        #     continue

        # reading the file containing the force data and the trajectories
        csv_path = os.path.join(csv_dir, file.split(".")[0] + ".csv")
        csv_dframe = pd.read_csv(csv_path)
        if len(csv_dframe.columns) < 2:
            csv_dframe = pd.read_csv(csv_path, delimiter=";")

        probs_in = np.squeeze(csv_dframe["Spinous Probability"])
        b_labels_in = np.squeeze(csv_dframe["Label"])
        step_labels = get_step_labels(list(b_labels_in))

        # smooth the probabilities
        probs = smooth(probs_in)

        # update the data_array with the pre-processed data

        # expanding one dimension to the labels to prepare them for concatenation
        counting_labels = np.expand_dims(step_labels, axis=1)
        binary_labels = np.expand_dims(b_labels_in, axis=1)
        probs = np.expand_dims(probs, axis=1)

        if not showFlag:
            continue

        if sacrum_flag:
            sacrum_probs = np.squeeze(csv_dframe["Sacrum"])
            sacrum_probs = smooth(sacrum_probs)
            sacrum_probs= np.expand_dims(sacrum_probs, axis=1)

            sacrum_labels = np.squeeze(csv_dframe["Sacrum_label"])
            sacrum_labels = np.expand_dims(sacrum_labels, axis=1)

            full_array = np.concatenate([probs, binary_labels, counting_labels,sacrum_probs,sacrum_labels], axis=1)
            sweep_name = str(file.split("_Spinous_Probability_label.csv")[0])
            # save the pre-processed data in the correct folder

            np.save(os.path.join(save_dir, sweep_name), full_array)

            n_rows = 5

            fig = plt.figure(figsize=(20, 10))
            fig.suptitle(file)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

            plt.subplot(n_rows, 1, 1)
            plt.plot(probs, linewidth=2)
            plt.title("probs")

            plt.subplot(n_rows, 1, 2)
            plt.plot(binary_labels, linewidth=2)
            plt.title("labels")

            plt.subplot(n_rows, 1, 3)
            plt.plot(counting_labels, linewidth=2)
            plt.title("labels_step")

            plt.subplot(n_rows, 1, 4)
            plt.plot(sacrum_probs, linewidth=2)
            plt.title("sacrum_prob")

            plt.subplot(n_rows, 1, 5)
            plt.plot(sacrum_labels, linewidth=2)
            plt.title("labels_sacrum")
            plt.show()

        else:
            # concatenate the data into one array
            full_array = np.concatenate([probs, binary_labels, counting_labels], axis=1)
            sweep_name = str(file.split("_Spinous_Probability_label.csv")[0])
            # save the pre-processed data in the correct folder

            if save:
                np.save(os.path.join(save_dir, sweep_name), full_array)
            n_rows = 3

            fig = plt.figure(figsize = (20, 10))
            fig.suptitle(file)

            plt.subplot(n_rows, 1, 1)
            plt.plot(full_array[:, 0], linewidth=2)
            plt.title("probs")

            plt.subplot(n_rows, 1, 2)
            plt.plot(full_array[:, -1], linewidth=2)
            plt.title("labels")

            plt.subplot(n_rows, 1, 3)
            plt.plot(full_array[:, -2], linewidth=2)
            plt.title("binary labels")
            plt.show()


if __name__ == "__main__":
    main()
