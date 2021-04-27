import numpy as np
import cv2


class GET2D:
    derivative_kernel = None
    smooth_kernel = None

    def __init__(self, derivative_kernel, smooth_kernel):
        self.derivative_kernel = derivative_kernel.astype(np.float64)
        self.smooth_kernel = smooth_kernel.astype(np.float64)

    def filter_image(self, image):

        image = image.astype(np.float64)

        gx = self.convolve_image(image, self.derivative_kernel, self.smooth_kernel)
        gy = self.convolve_image(image, self.smooth_kernel, self.derivative_kernel)

        gxx = self.convolve_image(gx, self.derivative_kernel, self.smooth_kernel)
        gxy = self.convolve_image(gx, self.smooth_kernel, self.derivative_kernel)

        gyy = self.convolve_image(gy, self.smooth_kernel, self.derivative_kernel)

        laplace = gxx + gyy

        gx3 = self.convolve_image(laplace, self.derivative_kernel, self.smooth_kernel)
        gy3 = self.convolve_image(laplace, self.smooth_kernel, self.derivative_kernel)

        get = self.get_full_tensor_image(gx, gy, gxx, gyy, gxy, gx3, gy3)
        return get

    @staticmethod
    def get_trace(tensor):
        get_trace = tensor[:, :, 0] + tensor[:, :, 2]
        return get_trace

    @staticmethod
    def convolve_image(image, kx, ky):

        if kx.shape[0] != 1:
            kx = np.transpose(kx)

        if ky.shape[1] != 1:
            ky = np.transpose(ky)

        conv1_result = cv2.filter2D(image, -1, kx, borderType=cv2.BORDER_REFLECT_101)

        conv2_result = cv2.filter2D(conv1_result, -1, ky, borderType=cv2.BORDER_REFLECT_101)
        return conv2_result

    @staticmethod
    def get_full_tensor_image(gx, gy, gxx, gyy, gxy, gx3, gy3):
        get = np.zeros([gx.shape[0], gx.shape[1], 3], dtype=np.float64)

        get[:, :, 0] = gxx ** 2 + gxy ** 2 - np.multiply(gx, gx3)
        get[:, :,  1] = - np.multiply(gxy, (gxx + gyy)) + 0.5 * (np.multiply(gx, gy3) + np.multiply(gy, gx3))
        get[:, :,  2] = gxy ** 2 + gyy ** 2 - np.multiply(gy, gy3)

        return get
