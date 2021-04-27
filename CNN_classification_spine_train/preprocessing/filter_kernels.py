import numpy as np
import cv2
import math


class GaussianDerivative:
    _sigma = 0.001
    _order = 0
    _function = None

    def __init__(self, sigma, order):
        self._sigma = sigma
        self._order = order

        sigma2 = -0.5 / (sigma ** 2)

        if order == 0:
            self._function = lambda x: 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(sigma2 * (x**2))

        elif order == 1:
            self._function = lambda x: -1 / (np.sqrt(2 * np.pi) * sigma**3) * np.exp(sigma2 * x**2) * x

        elif order == 2:
            self._function = lambda x: -1 / (np.sqrt(2 * np.pi) * sigma**3) * (1 - x ** 2 / sigma**2) * np.exp(sigma2 * (x**2))

        elif order == 3:
            self._function = lambda x: -1 / (np.sqrt(2 * np.pi) * sigma**5) * (2 - x ** 2 / sigma**2) * np.exp(sigma2 * (x**2))

    def at(self, x):
        """The returns the value of the Gaussian derivative evaluated in x

            :param x: The input value at which the Gaussian derivative is evaluated
            :returns: The Gaussian derivative evaluated in x
            """
        return self._function(x)

class Kernel1D:
    _kernel = None
    _left = 0
    _right = 0
    _norm = 0

    def normalize(self, norm, derivative_order):
        """The function normalizes the _kernel array, setting its norm to the value of the input argument "norm"

            :param norm: The desired norm value for the kernel
            :param derivative_order: The kernel derivative order.
            :param offset: ---
            """

        x = np.arange(self._left, self._left + len(self._kernel))
        kernel_sum = np.sum(self._kernel * np.power(-x, derivative_order) / math.factorial(derivative_order))

        self._kernel = self._kernel * norm / kernel_sum
        self._norm = norm

    def get_kernel(self):
        """The function returns the kernel array

            :returns: The kernel array
            """
        return self._kernel


class GaussianDerivativeKernel(Kernel1D):
    _sigma = 0
    _order = 0
    _radius = 0

    def __init__(self, sigma, order, norm=1):
        radius = int((3.0 + 0.5 * order) * sigma + 0.5)
        self._left = -radius
        self._right = radius
        self._radius = radius
        self._order = order
        self._sigma = sigma

        self._kernel = np.zeros(self._radius * 2 + 1, dtype=np.float64)
        self.create_kernel(norm)

    def create_kernel(self, norm):
        """The function creates the GaussianDerivative kernel with the radius specified in the self._radius attribute
        and with norm specified in the input argument norm. The generated kernel is stored in the self._kernel class
        attribute

            :param norm: The desired norm for the kernel.
            """

        g = GaussianDerivative(self._sigma, self._order)

        for i, x in enumerate(range(-self._radius, self._radius + 1)):
            self._kernel[i] = g.at(x)

        if self._order != 0:
            self._kernel = self._kernel - np.mean(self._kernel)

        if norm != 0:
            self.normalize(norm, self._order)
        else:
            self._norm = 1.0


class KernelGenerator:

    @staticmethod
    def gaussian_derivative(sigma, order):
        k = GaussianDerivativeKernel(sigma, order)
        return k.get_kernel()

    @staticmethod
    def gaussian(sigma, size):

        if len(size) < 1 or len(size) > 2:
            raise ValueError('only 1D and 2D Gaussian kernels can be computed')

        if size[0] == 1 or size[1] == 1:
            return cv2.getGaussianKernel(size[0] * size[1], sigma, cv2.CV_64F)

        kx = cv2.getGaussianKernel(size[0], sigma, cv2.CV_64F)
        ky = cv2.getGaussianKernel(size[0], sigma, cv2.CV_64F)
        return kx * np.transpose(ky)


