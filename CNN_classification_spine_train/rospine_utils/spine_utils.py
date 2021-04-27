import numpy as np
import cv2
from preprocessing import bone_probability_map_2d
import torch
import multiprocessing as mp


def _smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def _find_local_minima(input_array):

    local_mins = (np.diff(np.sign(np.diff(input_array))) > 0).nonzero()[0] + 1  # local min
    return local_mins


def _get_symmetry_metric(input_array, symmetry_point, delta):

    symmetry_metrics = 0
    for i in range(1, delta + 1):
        left_pt = input_array[symmetry_point - i]
        right_point = input_array[symmetry_point + i]

        symmetry_metrics += abs(left_pt - right_point)

    return -symmetry_metrics


def _compute_stream_mean_vector(img_list):
    mean_list = []
    for item in img_list:
        image = cv2.imread(item, 0)
        mean_list.append(np.sum(image))

    return np.array(mean_list)


def find_spinous_process_idx(img_list):

    mean_array = _compute_stream_mean_vector(img_list)
    smoothed_v, local_minima = _find_local_minima(mean_array)

    symmetry_list = []
    for min_point in local_minima:
        symmetry_list.append(_get_symmetry_metric(input_array=smoothed_v, symmetry_point=min_point, delta=30))

    most_symmetric_idx = np.argmax(symmetry_list)
    sp = local_minima[most_symmetric_idx]

    return sp


def apply_preprocessing(image, param):
    res = bone_probability_map_2d(image, param)

    min_val = np.min(res)
    max_val = np.max(res)
    res = (res - min_val) / (max_val - min_val)

    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # y
    min_val = np.min(sobely)
    max_val = np.max(sobely)
    sobely = (sobely - min_val) / (max_val - min_val)

    image3c = np.concatenate([np.expand_dims(image, axis=2), np.expand_dims(res * 255, axis=2),
                              np.expand_dims(sobely * 255, axis=2)], axis=2)

    return image3c


def get_sweep_crop(image_list, sp_idx, side):

    if side == 'right':
        return image_list[0:sp_idx]
    elif side == 'left':
        return image_list[sp_idx:-1]
    else:
        return None


def list2batch(tensor_list, label_list):

    if label_list is not None and len(label_list) == 0:
        label_list = None
    if label_list is not None:
        assert len(tensor_list) == len(label_list), "Number of labels files do not match number of images files"

    tensor_batch = torch.FloatTensor()
    torch.cat(tensor_list, out=tensor_batch)

    if label_list is None:
        return tensor_batch, None

    if type(label_list[0]) is torch.Tensor:
        label_batch = torch.FloatTensor()
        torch.cat(label_list, out=label_batch)

    else:
        label_batch = torch.FloatTensor(label_list)

    return tensor_batch, label_batch


class ThreadWrapper(object):
    thread_object = None

    def __init__(self):
        self.thread_object = mp.Process()

    def is_alive(self):
        return self.thread_object.is_alive()

    def start(self):
        return self.thread_object.start()

    def join(self):
        if self.thread_object.is_alive():
            self.thread_object.join()
        return

    def new(self, new_obj):
        self.thread_object = new_obj




