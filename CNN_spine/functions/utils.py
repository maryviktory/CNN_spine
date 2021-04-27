
import numpy as np
from scipy.interpolate import interp1d



def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

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


def interpolate_on_position(label_list, robot_pd):
    label_t = np.linspace(0, len(label_list) / 30, len(label_list))
    robot_time = robot_pd.loc[:, "timestamp"]
    robot_time = np.array(robot_time - robot_time[0])

    t_start, t_end = cut_force_frame(robot_time, label_t)
    # t_start, t_end = 1000, 2000

    robot_time = robot_time[t_start:t_end]

    f = interp1d(label_t, np.array(label_list))

    interp_label = f(robot_time)
    print(len(robot_time), "  ", len(interp_label))

    print(len(robot_pd.loc[t_start+1:t_end]))

    interp_label[interp_label > 0] = 1

    x_robot = np.array(robot_pd.loc[:, "Y"])
    print(x_robot[0], "  ---- ", x_robot[-1])

    x_robot = robot_pd.loc[t_start+1:t_end, "Y"]
    return x_robot, interp_label