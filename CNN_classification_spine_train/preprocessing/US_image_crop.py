import os
import SimpleITK as sitk
import numpy as np
import pandas as pd

root_dir = "/media/maria/Elements/to_NAS/VerseDbSimulation2"
file_dir = "/media/maria/Elements/to_NAS/VerseDbSimulation2/data_file.csv"

output_path = "/media/maria/Elements/to_NAS/VerseDbSimulation2_cropped"
batch_file_path = "/media/maria/Elements/to_NAS/VerseDbSimulation2_cropped/batch_file_path.txt"

target_dims = 250


def get_range(sum_array, target_dim):
    idx = np.squeeze(np.where(sum_array > 0))
    range_ = [idx[0], idx[-1]]
    size_ = idx[-1] - idx[0]
    if size_ < target_dim:
        rest = (target_dim - size_) % 2
        padding = int((target_dim - size_)/2)
        range_[0] = range_[0] - padding - rest
        range_[1] = range_[1] + padding
    elif size_ > target_dim:
        rest = (size_ - target_dim) % 2
        padding = (size_ - target_dim) / 2
        range_[0] = range_[0] + padding + rest
        range_[1] = range_[1] - padding
    return range_


fid = open(batch_file_path, 'w')
data_frame = pd.read_csv(file_dir)

fid.write("INPUTVOL;CROPLOWER;CROPUPPER;OUTPUTVOL")
for idx in range(113):

    inputvol = os.path.join(root_dir, data_frame.loc[idx, "volume"])
    inputlabel = os.path.join(root_dir, data_frame.loc[idx, "label"])
    print(idx, "  ", inputvol)

    outputvol = os.path.join(output_path, data_frame.loc[idx, "volume"])
    outputlabel = os.path.join(output_path, data_frame.loc[idx, "label"])

    image = sitk.ReadImage(inputvol)
    image_array = sitk.GetArrayFromImage(image)

    sum_2 = np.sum(np.sum(image_array, axis=0), axis=0) # 0, 1
    sum_1 = np.sum(np.sum(image_array, axis=0), axis=1) # 0, 2
    sum_0 = np.sum(np.sum(image_array, axis=1), axis=1) # 1, 2

    range_0 = get_range(sum_0, target_dims)
    range_1 = get_range(sum_1, target_dims)
    range_2 = get_range(sum_2, target_dims)

    x_range = [int(range_2[0]), int(image_array.shape[2] - range_2[1])]
    y_range = [int(range_1[0]), int(image_array.shape[1] - range_1[1])]
    z_range = [int(range_0[0]), int(image_array.shape[0] - range_0[1])]

    print(x_range)
    print(y_range)

    fid.write("\n" + inputvol + ";" + str(x_range[0]) + " " + str(y_range[0]) + " " + str(z_range[0]) + ";" + str(x_range[1]) + " " + str(y_range[1]) + " " + str(z_range[1]) + ";" + outputvol)
    fid.write("\n" + inputlabel + ";" + str(x_range[0]) + " " + str(y_range[0]) + " " + str(z_range[0]) + ";" + str(x_range[1]) + " " + str(y_range[1]) + " " + str(z_range[1]) + ";" + outputlabel)

fid.close()
