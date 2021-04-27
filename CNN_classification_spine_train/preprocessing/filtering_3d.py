import numpy as np
import cv2


def count_non_singleton_dimension(array):

    array_shape = array.shape
    nsd_numerosity = sum(1 for i in array_shape if i > 1)
    return nsd_numerosity


def filter1d_x(volume, kernel, border_type = cv2.BORDER_REFLECT_101):

    if len(volume.shape) != 3:
        raise ValueError("The input volume must be a 3D volume")

    if count_non_singleton_dimension(kernel) > 1:
        raise ValueError("The input kernel must be a 1D kernel")

    kx = np.reshape(kernel, [1, len(kernel)])
    filtered_volume = cv2.filter2D(volume, -1, kx, borderType=border_type)
    return filtered_volume


def filter1d_y(volume, kernel, border_type = cv2.BORDER_REFLECT_101):

    if len(volume.shape) != 3:
        raise ValueError("The input volume must be a 3D volume")

    if count_non_singleton_dimension(kernel) > 1:
        raise ValueError("The input kernel must be a 1D kernel")

    ky = np.reshape(kernel, [len(kernel), 1])
    filtered_volume = cv2.filter2D(volume, -1, ky, borderType=border_type)

    return filtered_volume


def filter1d_z(volume, kernel, border_type = cv2.BORDER_REFLECT_101):
    if len(volume.shape) != 3:
        raise ValueError("The input volume must be a 3D volume")

    if count_non_singleton_dimension(kernel) > 1:
        raise ValueError("The input kernel must be a 1D kernel")

    kz = np.reshape(kernel, [1, len(kernel)])

    rotated_volume = np.rot90(volume, k=1, axes=(1, 2))  # now z corresponds to the x axis, I filter over x
    filtered_volume = cv2.filter2D(rotated_volume, -1, kz, borderType=border_type)

    derotated_volume = np.rot90(filtered_volume, k=1, axes=(2, 1))
    return derotated_volume


def filter2d_yz(volume, kernel, border_type = cv2.BORDER_REFLECT_101):
    if len(volume.shape) != 3:
        raise ValueError("The input volume must be a 3D volume")

    if count_non_singleton_dimension(kernel) > 2:
        raise ValueError("The input kernel must be a 2D kernel")

    k_yz = np.squeeze(kernel)

    rotated_volume = np.rot90(volume, k=1, axes=(1, 2))
    filtered_volume = cv2.filter2D(rotated_volume, -1, k_yz, borderType=border_type)
    derotated_volume = np.rot90(filtered_volume, k=1, axes=(2, 1))
    return derotated_volume


def filter2d_xz(volume, kernel, border_type = cv2.BORDER_REFLECT_101):
    if len(volume.shape) != 3:
        raise ValueError("The input volume must be a 3D volume")

    if count_non_singleton_dimension(kernel) > 2:
        raise ValueError("The input kernel must be a 2D kernel")

    k_xz = np.squeeze(kernel)

    rotated_volume = np.rot90(volume, k=1, axes=(0, 2))
    filtered_volume = cv2.filter2D(rotated_volume, -1, k_xz, borderType=border_type)
    derotated_volume = np.rot90(filtered_volume, k=1, axes=(2, 0))
    return derotated_volume


def filter2d_xy(volume, kernel, border_type = cv2.BORDER_REFLECT_101):
    if len(volume.shape) != 3:
        raise ValueError("The input volume must be a 3D volume")

    if count_non_singleton_dimension(kernel) > 2:
        raise ValueError("The input kernel must be a 2D kernel")

    k_xy = np.squeeze(kernel)
    filtered_volume = cv2.filter2D(volume, -1, k_xy, borderType=border_type)
    return filtered_volume
