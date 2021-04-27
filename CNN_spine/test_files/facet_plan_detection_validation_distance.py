import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
import os
from scipy.interpolate import interp1d
from PIL import Image
import glob



def find_peaks(function,title,name,phase,bias = 0):
    # peaks - number of frame with local maximum
    # peak height
    if phase == 'robot':
        peaks, peak_height = signal.find_peaks(function[:1500], height=0,distance=50)
    else:
        peaks, peak_height = signal.find_peaks(function, height=0, distance=50)

    #sort from min to max the values of peaks
    sort_height_peak = np.argsort(peak_height.get('peak_heights'))
    #get the last three elements - highest peaks
    get_2_max_peaks = sort_height_peak[-2:]
    indexes_highest_peaks= []

    for i in range(len(get_2_max_peaks )):

        indexes_highest_peaks.append(peaks[get_2_max_peaks[i]])



    # index_first_peak = np.amin(indexes_highest_peaks)


    # plt.figure()
    # plt.suptitle('function_'+ title+name)
    # plt.plot(function)
    # indexes_highest_peaks = [item + bias for item in indexes_highest_peaks]
    # plt.plot(indexes_highest_peaks, function[indexes_highest_peaks], "x")
    return np.sort(indexes_highest_peaks)


def filter(y):
    # First, design the Buterworth filter
    N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, y)
    return filt

def interpolate_on_position(label_list, robot_list):
    label_t = np.linspace(0, len(label_list) / 30, len(label_list))
    robot_t = np.linspace(0,1, len(robot_list))

    f = interp1d(label_t, np.array(label_list))

    interp_label = f(robot_t)

    interp_label[interp_label > 0] = 1

    return interp_label

def image_intensity(image_folder):

    image_list = []
    mean_intensity = []
    delta = 100
    for filename in sorted(os.listdir(image_folder)):  # assuming gif

        im = Image.open(os.path.join(image_folder,filename))
        intensity = np.mean(im)
        image_list.append(im)
        mean_intensity.append(intensity)

    slice_inten = mean_intensity[(np.floor_divide(len(mean_intensity),2)-delta):(np.floor_divide(len(mean_intensity),2)+delta)]
    argmin = np.argmin(slice_inten)

    argmin = np.floor_divide(len(mean_intensity),2)-delta + argmin
    # argmin = np.argmin(mean_intensity)
    # print(argmin)
    # print(np.floor_divide(len(mean_intensity),2)-delta,(np.floor_divide(len(mean_intensity),2)+delta), 'args')

    return mean_intensity,argmin


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gausian_facet_enhansement(rob_function,probabilities, spinous_intensity_detected,atlas_sp_fac_distance_half):
    sig = 200
    between_facets_indeces_range = np.where((rob_function<rob_function[spinous_intensity_detected]+atlas_sp_fac_distance_half)&(rob_function>rob_function[spinous_intensity_detected]-atlas_sp_fac_distance_half))
    facet_1 = between_facets_indeces_range[0][0]
    facet_2 = between_facets_indeces_range[0][-1]
    print(facet_1,facet_2,'facets location')

    time_function = np.linspace(0, 1,len(probabilities))
    gaus1 = gaussian(time_function,facet_1,sig)
    gaus2 = gaussian(time_function,facet_2,sig)
    gaus = gaus1+gaus2

    # print(time_function)

    fused_function = gaus
    fused_function = np.multiply(probabilities,gaus)

    plt.figure()
    plt.plot(time_function,gaus)
    plt.plot(probabilities)
    plt.plot(fused_function)
    return fused_function






def main():
    CUTOFF = [0.1, 2]  # expresed in Hz
    FS = 1000  # sampling freqeuncy
    patients = []
    first_facet_array = []
    second_facet_array = []
    second_facet_dist = 0
    distance_between_facets_labels = []
    indeces_for_facet_distance = []
    indexes_highest_peaks = []
    gaussian_enhanced = []
    pd_frame = pd.DataFrame(columns=['sweep', 'Dist First Facet', 'Dist Second Facet'])

    path = "/media/maryviktory/My Passport/IPCAI 2020 TUM/Facet_Db/test/"
    path_image_folder = "/media/maryviktory/My Passport/IPCAI 2020 TUM/Facet_Db/test/images"
    save_path = "/media/maryviktory/My Passport/IPCAI 2020 TUM/Facet_Db/test"

    robot_traj_path = os.path.join(path, "robot_traj")
    files_in_robot_path = os.listdir(robot_traj_path)


    manual = True

    if manual == True:
        files_in_robot_path = []
        # good ones of all
        # list = [0,1,2,3,5,6,8,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,33]

        # 4 patients - 5 vertebrae each
        list = [0,1,2,3,4,7,8,9,14,15,19,20,21,22,23,26,27,28,29,30]
        # bad ones
        # list = [21,27,29]



        # list = [33,32,27,6,31,5,3,19]
        # list = [31,5]

        array_sweeps = np.array(list)
        for a in array_sweeps:
            files_in_robot_path.append('sweep%s.csv'%(a))


    for patient_file in files_in_robot_path:


        ####____Load_files_____###########

        patient = patient_file.split('.')[0]
        image_folder = os.path.join(path_image_folder, patient)
        path_prob_lab = os.path.join(path,"csv","%s_Facet_Probability_label.csv"%(patient))
        frame = pd.read_csv(path_prob_lab)
        robot_file = pd.read_csv(os.path.join(robot_traj_path,patient_file),sep=";")


        probability = np.array(frame.loc[:, "Facet Probability"])
        labels = np.array(frame.loc[:, "Label"])
        robot_traj = np.array(robot_file.loc[:, "y_1"])
        probability_filt = filter(probability)
        ####______find the interval which contains facet joint and peaks____#####

        mean_intensity,argmin_intensity = image_intensity(image_folder)
        distance_half = 24.2+10#with 10 tolerance

        flag = "manual"
        if flag == "manual":

            indeces_for_facet_distance = np.where((((robot_traj[argmin_intensity] - distance_half) <robot_traj) & ((robot_traj[argmin_intensity] +distance_half)> robot_traj)))
            indexes_highest_peaks_slice = find_peaks(probability_filt[indeces_for_facet_distance], "", "%s" % (patient),"")
            indexes_highest_peaks = indexes_highest_peaks_slice + indeces_for_facet_distance[0][0]
            # print(indeces_for_facet_distance,'indeces for facet distance')

        ######____Gaussian_ for facet enhancement and peaks find
        if flag == "gaussian":
            gaussian_enhanced = gausian_facet_enhansement(robot_traj,probability_filt,argmin_intensity,distance_half)
            indexes_highest_peaks = find_peaks(gaussian_enhanced,"","%s"%(patient),"")

        indeces_labels = np.sort(np.where(labels == 1)[0])
        first_facet_dist = np.abs(np.abs(robot_traj[indexes_highest_peaks[0]]) - np.abs(robot_traj[indeces_labels[0]] ))
        first_facet_array.append(first_facet_dist)


        print(patient)
        print(indeces_labels, 'labels')
        print(indexes_highest_peaks, 'indexes_highest_peaks')
        print(robot_traj[indexes_highest_peaks], "  robot_traj in probabilities peaks")
        print(robot_traj[indeces_labels], '  robot_traj in labels peaks')
        print('first facet dist', first_facet_dist)


        if (len(indeces_labels) & len(indexes_highest_peaks) )== 2:
            second_facet_dist = np.abs( np.abs(robot_traj[indexes_highest_peaks[1]]) - np.abs(robot_traj[indeces_labels[1]]))
            second_facet_array.append(second_facet_dist)
            distance_between_facets_labels.append(np.abs(np.abs(robot_traj[indeces_labels[0]]) - np.abs(robot_traj[indeces_labels[1]])))
            print('second facet dist', second_facet_dist)


        if flag == "manual":
            plt.figure()
            plt.title("__%s"%(patient_file))
            plt.xlabel('Frames',fontsize=15)
            plt.ylabel("CNN probabilities",fontsize=15)
            plt.plot(probability_filt, label = 'CNN probability')
            plt.plot(labels, 'r', label = 'Label')
            plt.plot(indexes_highest_peaks, probability_filt[indexes_highest_peaks], "x")

        if flag == "gaussian":
            plt.figure()
            plt.title("__%s" % (patient_file))
            plt.xlabel('Frames', fontsize=15)
            plt.ylabel("CNN probabilities", fontsize=15)
            plt.plot(gaussian_enhanced, label='CNN probability')
            plt.plot(labels, 'r', label='Label')
            plt.plot(indexes_highest_peaks, gaussian_enhanced[indexes_highest_peaks], "x")

        # plt.figure()
        # plt.plot(robot_traj, 'b', label='robot trajectory')
        # plt.plot(indexes_highest_peaks, robot_traj[indexes_highest_peaks], "x")
        # plt.legend(fontsize=15)

        # plt.figure()
        # plt.plot(mean_intensity,'r', label = 'mean intensity')
        # plt.plot(argmin_intensity,mean_intensity[argmin_intensity], "x")
        # plt.xlabel('Frames', fontsize=15)
        # plt.ylabel("Intensity", fontsize=15)
        # plt.title("intensity__%s" % (patient_file))
        # plt.plot(mean_intensity[indeces_for_facet_distance], "o")

        pd_frame = pd_frame.append({'sweep': patient, 'Dist First Facet': first_facet_dist, 'Dist Second Facet': second_facet_dist},
                                   ignore_index=True)

    pd_frame.to_csv(os.path.join(save_path,"Facet_Distance_4patients_5vert_each.csv"))



    print("First facet mean distance error is ", np.mean(first_facet_array), "with std: ", np.std(first_facet_array))
    print("Second facet mean distance error is ", np.mean(second_facet_array), "with std: ", np.std(second_facet_array))
    print("Mean distance between facets labels: ", np.mean(distance_between_facets_labels), "with std: ", np.std(distance_between_facets_labels))

    # plt.show()

main()