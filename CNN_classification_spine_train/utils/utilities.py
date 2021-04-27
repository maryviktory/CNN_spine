import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import SimpleITK as sitk


def normalize_01(input_array):
    tmp = (input_array - np.min(input_array))
    norm_array = tmp / np.max(tmp)
    return norm_array


def convert2uchar(image):
    uchar_image = cv2.normalize(np.float64(image), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    uchar_image = uchar_image.astype(np.uint8)
    return uchar_image


def show_3d(image, z_axis=0):

    if type(z_axis) != int:
        raise Exception("Invalid input z_axis type: z_axis must be int")

    if z_axis > 2:
        raise Exception("Invalid input z_axis type: z_axis must be in range [0, 2]")

    dims = len(image.shape)
    if dims < 2 or dims > 3:
        raise Exception("Invalid input image size: number of image axes should be equal to 2 or 3")

    uchar_image = convert2uchar(image)

    plt.ioff()
    if dims == 2:
        plt.imshow(uchar_image, cmap='gray', vmin=0, vmax=255)
        plt.imshow()
        return

    plt.ion()
    squeezed_image = None
    for i in range(image.shape[z_axis]):
        if z_axis == 0:
            squeezed_image = np.squeeze(uchar_image[i, :, :])
        elif z_axis == 1:
            squeezed_image = np.squeeze(uchar_image[:, i, :])
        else:
            squeezed_image = np.squeeze(uchar_image[:, :, i])

        plt.imshow(squeezed_image, cmap='gray', vmin=0, vmax=255)
        plt.pause(0.03)
        plt.clf()

    plt.ioff()
    plt.imshow(squeezed_image, cmap='gray', vmin=0, vmax=255)
    plt.show()


def select_roi(image):

    uchar_image = convert2uchar(image)
    rgb_image = cv2.cvtColor(uchar_image, cv2.COLOR_GRAY2BGR)
    roi = cv2.selectROI("Roi selection", rgb_image)  # x, y, w, h
    return roi


def label_images(folder_path, show=False):

    df = pd.DataFrame(columns=["image_name", "x", "y", "z", "w", "h", "d"])
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.mhd')]

    for i, file in enumerate(file_list):

        file_path = os.path.join(folder_path, file)
        itk_image = sitk.ReadImage(file_path)
        volume = sitk.GetArrayFromImage(itk_image)  # order z, y, x

        xy_projection = np.mean(volume, axis=1)
        xz_projection = np.mean(volume, axis=0)

        roi1 = select_roi(xy_projection)  # x, y, w, h
        roi2 = select_roi(xz_projection)  # z, y, d, h

        x, y, z, w, h, d = roi1[0], roi1[1], roi2[0], roi1[2], roi1[3], roi2[2]

        cropped_image = volume[y:y+h, x:x+w, z:z+d]
        df.loc[i] = [file, x, y, z, w, h, d]

        if show:
            show_3d(cropped_image, z_axis=1)

    return df


def show_slices(volume, plane = "xy"):

    k = 0
    while True:
        slice_to_show = np.ones([volume.shape[0] + 40, volume.shape[1]]) * 255
        volume_slice = np.squeeze(volume[:, :, k])
        slice_to_show[0:-40, :] = volume_slice
        cv2.putText(slice_to_show, "Slice" + str(k), (20, volume.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0)

        cv2.imshow("volume", slice_to_show)

        button = cv2.waitKey(33)
        if button == ord('u'):
            k = k + 1
        elif button == ord('d'):
            k = k-1
        elif button == ord('q'):
            break

        if k < 0:
            k = 0
        if k > volume.shape[2] - 1:
            k = volume.shape[2] - 1

    cv2.destroyAllWindows()


def save_itk(filename, image, meta_data = None):
    itkimage = sitk.GetImageFromArray(image, isVector=False)

    if meta_data is not None and type(meta_data) == dict:
        if "spacing" in meta_data:
            itkimage.SetSpacing(meta_data["spacing"])
        if "origin" in meta_data:
            itkimage.SetOrigin(meta_data["origin"])
        if "direction" in meta_data:
            itkimage.SetDirection(meta_data["direction"])


    #itkimage.SetPixel(sitk.sitkLabelUInt8)
    sitk.WriteImage(itkimage, filename, True)


def read_stk(filename):
    itk_image = sitk.ReadImage(filename)

    meta_data = dict()
    meta_data["origin"] = itk_image.GetOrigin()
    meta_data["spacing"] = itk_image.GetSpacing()
    meta_data["direction"] = itk_image.GetDirection()

    image = sitk.GetArrayFromImage(itk_image)  # order z, y, x

    return image, meta_data

