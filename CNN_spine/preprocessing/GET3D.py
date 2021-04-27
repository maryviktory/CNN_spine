import numpy as np
import preprocessing


class GET3D:
    derivative_kernel = None
    smooth_kernel = None

    def __init__(self, derivative_kernel, smooth_kernel):
        self.derivative_kernel = derivative_kernel
        self.smooth_kernel = smooth_kernel

    def convolve_image(self, image, derivative_axis):

        if derivative_axis == "x":  # derivative is taken over the x axis, smoothing on the other 2
            x_derivative = preprocessing.filter1d_x(image, self.derivative_kernel)
            yz_smoothing = preprocessing.filter2d_yz(x_derivative, self.smooth_kernel)
            return yz_smoothing

        elif derivative_axis == "y":  # derivative is taken over the y axis, smoothing on the other 2
            y_derivative = preprocessing.filter1d_y(image, self.derivative_kernel)
            xz_smoothing = preprocessing.filter2d_xz(y_derivative, self.smooth_kernel)
            return xz_smoothing

        elif derivative_axis == "z":  # derivative is taken over the z axis, smoothing on the other 2
            z_derivative = preprocessing.filter1d_z(image, self.derivative_kernel)
            xy_smoothing = preprocessing.filter2d_xz(z_derivative, self.smooth_kernel)
            return xy_smoothing

        else:
            print("derivative axis exceeds matrix dimensions")
            return None

    @staticmethod
    def get_even_part(gxx, gyy, gzz, gxy, gxz, gyz):
        get_even = np.zeros([gxx.shape[0], gxx.shape[1], gxx.shape[2], 6], dtype=np.float64)

        get_even[:, :, :, 0] = gxx ** 2 + gxy ** 2 + gxz ** 2 # T_11
        get_even[:, :, :, 1] = np.multiply(gxy, gxx) + np.multiply(gyy, gxy) + np.multiply(gyz, gxz) # T_21 = T_12
        get_even[:, :, :, 2] = gxy**2 + gyy**2 + gyz**2 # T_22
        get_even[:, :, :, 3] = np.multiply(gxz, gxx) + np.multiply(gyz, gxy) + np.multiply(gzz, gxz) # T_31
        get_even[:, :, :, 4] = np.multiply(gxz, gxy) + np.multiply(gyz, gyy) + np.multiply(gzz, gyz) # T_32
        get_even[:, :, :, 5] = gxz**2 + gyz**2 + gzz**2 # T_33

        return get_even

    @staticmethod
    def get_odd_part(gx, gy, gz, gx3, gy3, gz3):
        get_odd = np.zeros([gx.shape[0], gx.shape[1], gx.shape[2], 6], dtype=np.float64)

        get_odd[:, :, :, 0] = - np.multiply(gx, gx3) # T_11
        get_odd[:, :, :, 1] = - 0.5 * (np.multiply(gx3, gy) + np.multiply(gy3, gx)) # T_21 = T_12
        get_odd[:, :, :, 2] = - np.multiply(gy3, gy) # T_22
        get_odd[:, :, :, 3] = - 0.5 * (np.multiply(gx3, gz) + np.multiply(gz3, gx)) # T_31 = T_13
        get_odd[:, :, :, 4] = - 0.5 * (np.multiply(gy3, gz) + np.multiply(gz3, gy)) # T_32 = T_23
        get_odd[:, :, :, 5] = - np.multiply(gz3, gz) # T_33

        return get_odd

    def filter_image(self, image):
        gx = self.convolve_image(image, "x")
        gy = self.convolve_image(image, "y")
        gz = self.convolve_image(image, "z")

        gxx = self.convolve_image(gx, "x")
        gxy = self.convolve_image(gx, "y")
        gxz = self.convolve_image(gx, "z")
        gyz = self.convolve_image(gy, "z")

        gyy = self.convolve_image(gy, "y")
        gzz = self.convolve_image(gz, "z")

        laplace = gxx + gyy + gzz

        gx3 = self.convolve_image(laplace, "x")
        gy3 = self.convolve_image(laplace, "y")
        gz3 = self.convolve_image(laplace, "z")

        get_even = self.get_even_part(gxx, gyy, gzz, gxy, gxz, gyz)
        get_odd = self.get_odd_part(gx, gy, gz, gx3, gy3, gz3)
        return get_even, get_odd