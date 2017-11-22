import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from scipy.signal import convolve2d as convolve2d

from skimage.color import rgb2gray


def read_image(filename, representation):
    """
    :param fileame: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be
    a grayscale image (1) or an RGB image (2).
    """
    im = imread(filename, mode="RGB")
    if (representation == 1 and im.ndim > 2):
        im = rgb2gray(im)
    elif (representation == 1):
        im = (im / 255).astype(np.float64)
        # pass
    elif (representation == 2):
        im = (im / 255).astype(np.float64)
    else:
        print("Error")

    # im = im / 255
    return (im)


def dft_matrix(N):
    """:return the n*n dft matrix"""
    exp_const = -np.pi * 2j / N
    matrix = np.fromfunction(lambda x, u: np.exp(exp_const * (x * u)), (N, N), dtype=np.complex128)
    return matrix


def idft_matrix(N):
    """:return the n*n idft matrix"""
    exp_const = np.pi * 2j / N
    matrix = np.fromfunction(lambda x, u: np.exp(exp_const * (x * u)), (N, N), dtype=np.complex128)
    return matrix


def DFT(signal):
    """transform a 1D discrete signal to its Fourier representation"""
    return np.dot(dft_matrix(signal.shape[0]), signal)


def IDFT(fourier_signal):
    """ performs inverse DFT - transform a 1D Fourier representation to discrete signal """

    N = len(fourier_signal)
    # print(N)
    return np.dot(idft_matrix(fourier_signal.shape[0]), fourier_signal / N)


def DFT2(image):
    """convert a 2D discrete signal to its Fourier representation"""
    rows, cols = image.shape
    tmp_rows = np.dot(dft_matrix(rows), image)

    return np.dot(tmp_rows, dft_matrix(cols))


def IDFT2(fourier_image):
    """performs inverse DFT - convert a 2D Fourier representation to discrete signal """
    rows, cols = fourier_image.shape
    tmp_rows = np.dot(idft_matrix(rows), fourier_image)

    return np.dot(tmp_rows, idft_matrix(cols)) / (rows * cols)


def conv_der(im):
    """computes the magnitude of image derivatives"""
    conv = np.array([[1, 0. - 1]]) ## TODO * 0.5
    dx = convolve2d(im, conv, mode="same")
    conv = conv.T
    dy = convolve2d(im, conv, mode="same")
    # plt.imshow(dy, cmap="gray")

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    # print(magnitude)
    return magnitude


def fourier_der(im):
    """computes the magnitude of image derivatives using Fourier transform"""
    rows, cols = im.shape
    exp_const_x = np.pi * 2j / rows
    exp_const_y = np.pi * 2j / cols
    F = np.fft.fftshift(DFT2(im))
    U_s = np.fromfunction(lambda u, v: u - (rows // 2), (rows, cols)) * exp_const_x

    # print(U_s)
    dx = IDFT2(np.multiply(F, U_s))
    V_x = np.fromfunction(lambda u, v: v - (cols // 2), (rows, cols)) * exp_const_y
    dy = IDFT2(np.multiply(F, V_x))
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def calc_kernel(kernel_size):
    """returns approximation of the gaussian kernel using the binomial coefficients
    with shape (kernel_size, kernel_size)in image space """
    if (kernel_size == 1):
        return np.array([[1]])
    base_kernel = np.array([1, 1])
    kernel = np.array([1, 1],np.float64)
    for i in range(kernel_size - 2):
        kernel = np.convolve(kernel, base_kernel)
    kernel = kernel / kernel.sum()
    # a = np.pad(kernel, ((kernel_size // 2, kernel_size // 2), (0, 0)), mode="constant")
    # b = np.pad(kernel.T, ((0, 0), (kernel_size // 2, kernel_size // 2)), mode="constant")
    # print("______________________")
    kernel = np.matrix([kernel])
    kernel = kernel.T*kernel

    # kernel_sum = kernel.sum()
    # print(kernel)
    # kernel = kernel / kernel_sum

    return kernel


def blur_spatial(im, kernel_size):
    """performs image blurring using 2D convolution between the image im and a gaussian
    kernel with size kernel_size*kernel_size."""
    kernel = calc_kernel(kernel_size)
    # print(kernel)
    return convolve2d(im, kernel, mode="same")
    # print (b)
    # print(np.add(a,b))


def blur_fourier(im, kernel_size):
    """performs image blurring with gaussian kernel in Fourier space."""
    rows, cols = im.shape
    kernel = calc_kernel(kernel_size)
    k_r, k_c = kernel.shape
    # s_kernel = np.zeros(im.shape)
    # s_kernel[cols//2-k_r//2:cols//2+k_r][rows//2-k_] = kernel
    r = not rows % 2
    c = not cols % 2
    padded_kernel = np.pad(kernel, (((rows - k_r) // 2, (rows - k_r) // 2 + r),
                                    ((cols - k_c) // 2, (cols - k_c) // 2 + c)), mode="constant")
    # print("im =  ", im.shape)
    # print("kernel = ", padded_kernel.shape)
    kernel_F = DFT2(padded_kernel)
    im_F = DFT2(im)
    blur_F = np.multiply(im_F, kernel_F)
    return np.real(np.fft.ifftshift(IDFT2(blur_F)))
#
# print(calc_kernel(1).shape)
# add = "../Ex1/ss.jpg"
# my_pic = read_image(add, 1)
# my = blur_fourier(my_pic,11)
# my = fourier_der(my)
#
# plt.imshow(my,cmap="gray")
# plt.show()

# bs = blur_fourier(my_pic, 31)
# print(bs)
# bs = np.real_if_close(bs, 1000000)
# c = conv_der(my_pic)
# f = fourier_der(my_pic)
# plt.imshow(f, cmap="gray")
# plt.show()
# plt.imshow(c, cmap="gray")
# plt.show()
# a = dft_matrix(5)
# b = np.asarray([5, 1, 2, 3, 4]).reshape(5, 1)
# print(a)
# print(b)
# print(IDFT(DFT(b)))
# my_im = np.asarray(range(30)).reshape(5,6)
# for i in my_im:
#     print (i)
# my = IDFT2(DFT2(my_pic))
# dft1 = DFT2(my_pic)
#
# dft2 = IDFT2(dft1)
# dft2 = np.real_if_close(dft2,100000)
# plt.imshow(dft2,cmap="gray")
# plt.show()
#
#
# nps = np.fft.ifft2(np.fft.fft2(my_pic))
#
# print(np.allclose(my,nps))
#
