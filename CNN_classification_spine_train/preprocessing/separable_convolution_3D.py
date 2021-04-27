import numpy as np
import cv2

# NOT USED AT THE MOMENT

def flatten2d(volume, split_axis, concatenation_axis):
    split_array = np.split(volume, volume.shape[split_axis], axis=split_axis)
    flatten_array = np.concatenate(split_array, axis=concatenation_axis)
    return np.squeeze(flatten_array)


def image2volume(image, num_channels, split_axis, concatenation_axis):
    split_image = np.split(image, num_channels, axis=split_axis)
    array3d = np.stack(split_image, axis=concatenation_axis)
    return array3d


def squeeze_channels_in_rows(volume):
    return flatten2d(volume, 2, 0)


def squeeze_channels_in_cols(volume):
    return flatten2d(volume, 2, 1)


def squeeze_cols_in_rows(volume):
    return flatten2d(volume, 1, 0)


def split_rows_in_channels(image, n_channels):
    return image2volume(image, n_channels, 0, 2)


def split_cols_in_channels(image, n_channels):
    return image2volume(image, n_channels, 1, 2)


def filter_over_rows(image, kernel, border_type=cv2.BORDER_REFLECT_101):
    kernel = np.squeeze(kernel)
    kernel = np.expand_dims(kernel, axis=0)

    conv_result = cv2.filter2D(image, -1, kernel, borderType=border_type)

    return conv_result


def filter_over_cols(image, kernel, border_type=cv2.BORDER_REFLECT_101):
    kernel = np.squeeze(kernel)
    kernel = np.expand_dims(kernel, axis=1)

    conv_result = cv2.filter2D(image, -1, kernel, borderType=border_type)

    return conv_result


def sepconv3d_x(volume, kernel):
    n_channels = volume.shape[2]
    flatten_volume = squeeze_channels_in_rows(volume)
    filtered_rows = filter_over_rows(flatten_volume, kernel)
    filtered_volume = split_rows_in_channels(filtered_rows, n_channels)
    return filtered_volume


def sepconv3d_y(volume, kernel):
    n_channels = volume.shape[2]
    flatten_volume = squeeze_channels_in_cols(volume)
    filtered_rows = filter_over_cols(flatten_volume, kernel)
    filtered_volume = split_cols_in_channels(filtered_rows, n_channels)
    return filtered_volume


def sepconv3d_z(volume, kernel):
    n_channels = volume.shape[1]
    flatten_volume = squeeze_cols_in_rows(volume)
    filtered_rows = filter_over_rows(flatten_volume, kernel)
    filtered_volume = split_rows_in_channels(filtered_rows, n_channels)
    rotated_volume = np.rot90(filtered_volume, 1)
    return rotated_volume


def separable_convolution3d(volume, kx, ky, kz):
    filtered_volume_x = sepconv3d_x(volume, kx)
    filtered_volume_xy = sepconv3d_y(filtered_volume_x, ky)
    filtered_volume_xyz = sepconv3d_z(filtered_volume_xy, kz)
    return filtered_volume_xyz


