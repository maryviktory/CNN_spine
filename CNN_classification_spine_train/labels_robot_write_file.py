### FIND the probabilities of Classes, "Gap", "NonGap"
### p_c1 - non Gap
### p_c2 - Gap

import rospine_utils as utils
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas
import os
from scipy.interpolate import interp1d

# patient_name = 'Maria_V'
# sweep_folder = "/media/maria/My Passport/toNas/data set patients images/%s/Images"%patient_name
# sweep_label_folder = "/media/maria/My Passport/toNas/data set patients images/%s/Labels"%patient_name
# save_folder = "/media/maria/My Passport/toNas/data set patients images/%s/"%patient_name
def cut_force_frame(timeseries_force,timeseries_images):
    time = 0
    i = 0

    for i in range(0, len(timeseries_force)):
        if time <= max(timeseries_images) - 0.1:
            time = timeseries_force[i]
            # print(i)


        else:
            break
        # i = i + 1

    end_frame_cut = i
    print('end force time cut', timeseries_force[end_frame_cut])

    time = 0
    k = 0
    # print(min(timeseries_images))

    for k in range(0, len(timeseries_force)):

        if time <= min(timeseries_images):
            time = timeseries_force[k]

        else:
            break

        # k = k + 1
        # print(i)
        # print(time)
    print(k)
    start_time_force_cut = k

    return start_time_force_cut, end_frame_cut


# def interpolate_on_position(label_list, robot_pd):
#     label_t = np.linspace(0, len(label_list) / 30, len(label_list))
#     robot_time = robot_pd.loc[:, "timestamp"]
#     robot_time = np.array(robot_time - robot_time[0])
#
#     t_start, t_end = cut_force_frame(robot_time, label_t)
#     # t_start, t_end = 1000, 2000
#
#     robot_time = robot_time[t_start:t_end]
#
#     f = interp1d(label_t, np.array(label_list))
#
#     interp_label = f(robot_time)
#     print(len(robot_time), "  ", len(interp_label))
#
#     print(len(robot_pd.loc[t_start+1:t_end]))
#
#     interp_label[interp_label > 0] = 1
#
#     x_robot = np.array(robot_pd.loc[:, "Y"])
#     print(x_robot[0], "  ---- ", x_robot[-1])
#
#     x_robot = robot_pd.loc[t_start+1:t_end, "Y"]
#     return x_robot, interp_label

def interpolate_on_position(label_list, robot_pd):
    label_t = np.linspace(0, len(label_list) / 30, len(label_list))
    robot_time = robot_pd.loc[:, "timestamp"]
    robot_time = np.array(robot_time - robot_time[0])

    # t_start, t_end = cut_force_frame(robot_time, label_t)
    # t_start, t_end = 1000, 2000

    # robot_time = robot_time[t_start:t_end]

    f = interp1d(label_t, np.array(label_list))

    interp_label = f(robot_time)

    print(len(robot_time), " len robot__len__image", len(interp_label))

    # print(len(robot_pd.loc[t_start+1:t_end]))

    interp_label[interp_label > 0] = 1

    x_robot = np.array(robot_pd.loc[:, "Y"])
    print(x_robot[0], "  ---- ", x_robot[-1])

    # x_robot = robot_pd.loc[t_start+1:t_end, "Y"]
    return x_robot, interp_label

def main():
    name = "Ardit"

    sweep_dir = '/media/maryviktory/My Passport/IPCAI 2020 TUM/Force_integration_DB/images'
    sweep_label_dir = '/media/maryviktory/My Passport/IPCAI 2020 TUM/Force_integration_DB/labels'
    robotic_trajectory = "/media/maryviktory/My Passport/IPCAI 2020 TUM/Force_integration_DB/robot_traj"


    # save_folder = '/media/maria/My Passport/SpinousProcessDb'

    # force_modes = ["_F10", "_F15", "_F2"]
    force_modes = [ "_F2"]
    plt.figure()

    for i, force_mode in enumerate(force_modes):

        frame_dir = os.path.join(robotic_trajectory, name+force_mode + ".csv")
        sweep_folder = os.path.join(sweep_dir, name + force_mode)
        sweep_label_folder = os.path.join(sweep_label_dir, name + force_mode)

        print(frame_dir)
        print(sweep_folder)
        print(sweep_label_folder)

        pd_frame = pandas.read_csv(frame_dir)
        print(pd_frame.columns)
        print()

        loader = utils.OfflineLoader(sweep_path=sweep_folder,
                                     sweep_label_path=sweep_label_folder,
                                     sort_type=0,
                                     bias=0)

        # Initializing processing thread
        label_list = []


        while True:

            current_image, current_label = loader.get_next_image()
            if current_image is None:
                break

            label_list.append(current_label)

        print(len(label_list))


        plt.subplot(3, 1, i + 1)
        x_img = np.array(pd_frame.loc[:, "Y"])
        plt.plot(x_img, label_list)

        x_robot, label_robot = interpolate_on_position(np.array(label_list), pd_frame)

        plt.plot(x_robot)
        plt.plot(label_robot)

        # x_img = np.array(pd_frame.loc[:, "X"])
        # z_img = np.array(pd_frame.loc[:, "Z"])
        #
        #
        # ax1 = plt.subplot(3, 1, 1)
        # ax1.set_title("X axis filt")
        # ax1.plot(x_img)
        # #
        # ax2 = plt.subplot(3, 1, 2)
        # ax2.set_title("Y axis filt")
        # ax2.plot(y_img)
        #
        # ax3 = plt.subplot(3, 1, 3)
        # ax3.set_title("Z axis filt")
        # ax3.plot(z_img)


    plt.show()

main()

