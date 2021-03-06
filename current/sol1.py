import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray

RGB2YIQ_MATRIX = np.array([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


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


def imdisplay(filename, representation):
    """
    The function will open a new figure and display the loaded image in the converted
    representation.
    """
    im = read_image(filename, representation)
    plt.imshow(im, cmap="gray")
    plt.show()


def rgb2yiq(imRGB):
    """
    transform an RGB image into the YIQ color space.
    :param imRGB:
    :return: height × width × 3 np.float64 matrix with values in [0, 1].
    """

    yiq_im = np.dot(imRGB.copy(), RGB2YIQ_MATRIX.copy().transpose())
    return yiq_im


def yiq2rgb(imYIQ):
    """
    transform an YIQ image into the RGB color space.
    :param imYIQ: height × width × 3 np.float64 matrix with values in [0, 1].
    :return:
    """
    rgb_im = np.dot(imYIQ.copy(), np.linalg.inv(RGB2YIQ_MATRIX.copy()).transpose())
    return rgb_im


def histogram_equalize(im_orig):
    """
    :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
    :return: The function returns a list [im_eq, hist_orig, hist_eq] where
    im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
    hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
    hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    dims = im_orig.ndim
    if (dims >2):
        img = rgb2yiq(im_orig)
        im = img.copy()[:,:,0]
    else:
        im = im_orig.copy()
    im = (im * 255).astype(np.uint8)
    hist_orig = np.histogram(im,256,range = (0,255))[0]
    cumulative_hist = np.cumsum(hist_orig)
    m = np.argmax(cumulative_hist>0)
    lut = np.asarray([((k-cumulative_hist[m])*255/(cumulative_hist[-1]-cumulative_hist[m])) for k in cumulative_hist])\
        .astype(np.uint8)

    im_eq = lut[im]##np.interp(im.flatten(),range(256), norm_cumulative_hist).reshape (im.shape)
    hist_eq =  np.histogram(im_eq,256,range = (0,255))[0]

    im_eq = im_eq / 255
    if (dims>2):
        ##TODO :: CHECK /255 ISSUES (RGB/GREY)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ##TODO :: clip the iimage
        img[:, :, 0] = im_eq
        im_eq = np.clip(yiq2rgb(img),0,1)

    return [im_eq, hist_orig, hist_eq]

def quantize (im_orig, n_quant, n_iter):
    """
    :param im_orig: grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: number of intensities the output image should have.
    :param n_iter: maximum number of iterations of the optimization procedure
    :return: im_quant - is the quantized output image
             error - is an array with the total intensities error for each iteration of the
             quantization procedure.
    """

    dims = im_orig.ndim
    if (dims > 2):
        img = rgb2yiq(im_orig)
        im = img.copy()[:, :, 0]
    else:
        im = im_orig.copy()
    im = (im * 255).astype(np.uint8)
    z = np.asarray([0]* (n_quant+1))
    q = np.asarray([0.0 for i in range (n_quant)])
    hist = np.histogram(im,256,range = (0,255))[0]
    cumulative_hist =np.cumsum(hist)
    max_per_slot = cumulative_hist[-1] // n_quant
    z[0] = 0

    for i in range (n_quant):
        z[i] = np.argmax(cumulative_hist>=max_per_slot*i)
    z[0] = 0.0
    z [-1] = 255
    error = []
    cur_iter = 0

    while (cur_iter < n_iter):

        err_calc = [0.0]*256

        q[0] = np.sum(np.asarray(range(z[0],z[1] + 1)).dot(hist[z[0]:z[1] + 1])) / np.sum(
                hist[z[0]:z[1] + 1])

        q[1:]=np.asarray([(np.dot(np.asarray(range(z[i]+1,z[i+1]+1)),hist[z[i]+1:z[i+1]+1])) /
                           np.sum(hist[z[i]+1:z[i+1]+1]) for i in range(1,len(q))])
        for i in range(n_quant):
            err_calc[z[i]:z[i + 1]] = [q[i]] * (z[i + 1] - z[i])
        err_calc = np.square(np.subtract(np.asarray(range(0,256)),err_calc))
        error.append(np.dot(hist,err_calc))
        z_old = z.copy()
        z[1:-1] = np.asarray([(q[i-1]+q[i])//2 for i in range(1, len(q))])

        cur_iter +=1
        if (np.array_equal(z_old , z)):
            break

    quant_hist =[0]*256
    for i in range(n_quant):
        quant_hist[z[i]:z[i+1]] = [q[i]]*(z[i+1]-z[i])
    quant_hist[-1] = q[-1]
    quant_hist = np.asarray(quant_hist,np.uint8)
    im = quant_hist[im]

    im = im / 255
    if (dims>2):
        img[:, :, 0] = im
        im_quant = yiq2rgb(img)
    else:
        im_quant = im

    return [im_quant, error]

# pic_add = "original.png"
# a = "equlized.png"
# eq_by_school = read_image(a, 1)
# eq_by_school = eq_by_school.astype(np.uint)
# th = np.histogram(eq_by_school,256,range=(0,255) )[0]
# im = read_image(pic_add, 2)
# plt.imshow(im,cmap = "gray")
# plt.show()
# x,z,y=histogram_equalize(im)
# print ("______________________________")
# print (z)
#
#
# im_quant,err = quantize (im,15,50)
# print (err)
# plt.plot(err)
# plt.show()
#
# im = (yiq2rgb(rgb2yiq(im)))
# print (im_quant)
# plt.imshow(np.clip(im_quant,0,1),cmap = "gray")
# plt.show()
# # print (np.max(np.subtract(im,imk)))
#
# print (im.flatten())
# print(im)
# im_eq,hist_orig,hist_eq= histogram_equalize(im)
# plt.plot(hist_eq)
# plt.plot(th)
# plt.show()
# plt.imshow(np.clip(im_eq,0,1),cmap = "gray")
# plt.show()
# print (hist_eq)
# rgb2yiq(read_image(pic_add, 2))
# print (im.shape)
# imdisplay(pic_add, 1)
# for item in RGB2YIQ_MATRIX:
#     print(item)
# print(yiq2rgb(rgb2yiq(im)).shape)
#
# print(im.shape[0]*im.shape[1])
# print(im.dtype)
# print (im)
