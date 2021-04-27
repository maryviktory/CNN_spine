import numpy as np
import cv2
import preprocessing
import utils
import scipy.ndimage as scind
import time
import multiprocessing
import matplotlib.pyplot as plt

from scipy.linalg import hadamard


def compute_scanlines(volume, param):
    # the function should return a volume in which each channel correspond to an acquired frame and each column in the
    # channel to a scanline for that specific frame
    return volume


def compute_rn(volume, param, return_dict, dtype=np.float64):
    if dtype(volume) != dtype:
        volume = volume.astype(dtype)

    lapl_of_gauss = scind.gaussian_laplace(volume, sigma = param["LoG"]["sigmaXYZ"])
    lapl_of_gauss[lapl_of_gauss > 0] = 0
    lapl_of_gauss = -lapl_of_gauss

    blurred_image = scind.gaussian_filter(volume, sigma=param["Blurr"]["sigmaXYZ"])
    rn_image = lapl_of_gauss + blurred_image

    return_dict["rn"] = rn_image


def compute_sh(volume, param, return_dict = None):

    x_array = np.arange(0, volume.shape[0])
    g = preprocessing.GaussianDerivative(param["GaussSigma"], 0)
    gauss_array = g.at(x_array)

    padded_vol = cv2.copyMakeBorder(volume, top=0, bottom=int(len(gauss_array)/2), left=0, right=0,
                                    borderType=cv2.BORDER_CONSTANT, value=0)

    filter_res = cv2.filter2D(padded_vol, -1, kernel = gauss_array, borderType=cv2.BORDER_ISOLATED)

    numerator = filter_res[int(len(gauss_array) / 2)::, :]
    denominator = np.zeros(volume.shape)
    ones_vol = np.ones([1, volume.shape[1]])

    for i in range(len(denominator)):
        if i == 0:
            denominator[i, :] = 1 / np.sum(gauss_array[0::]) * ones_vol
        else:
           denominator[i, :] = 1 / np.sum(gauss_array[0:-i]) * ones_vol

    return_dict["sh"] = np.multiply(numerator, denominator)


def bone_probability_map_2d(volume, param):

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    scanlines = compute_scanlines(volume, None)

    p1 = multiprocessing.Process(target=compute_rn, args=(volume,param, return_dict))
    p2 = multiprocessing.Process(target=compute_sh, args=(scanlines, param, return_dict))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    rn_norm = utils.normalize_01(return_dict["rn"])
    sh_n = utils.normalize_01(return_dict["sh"])

    bpm = rn_norm + sh_n
    return bpm


