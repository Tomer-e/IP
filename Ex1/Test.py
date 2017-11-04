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
    im = imread(filename)
    # print (im)
    im = im.astype(np.float64)
    if (representation == 1 and im.ndim > 2):
        im = rgb2gray(im)
    elif (representation == 1):
        pass
    elif (representation == 2):
        pass
    else:
        print("Error")

    im = im / 255
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
    # imRGB = np.asanyarray(imRGB)
    # redIm = imRGB[:,:,0]
    # greenIm = imRGB[:,:,1]
    # blueIm = imRGB[:,:,2]
    # redIm *= RGB2YIQ_MATRIX[0][0] # 0.299
    # blueIm *= RGB2YIQ_MATRIX[0][1]
    # greenIm *= RGB2YIQ_MATRIX[0][2]
    # imRGB[:,:,0] = redIm + blueIm + greenIm
    # imRGB[:,:,1] = redIm + blueIm + greenIm
    # imRGB[:,:,2] = redIm + blueIm + greenIm
    # np.dot(imRGB,RGB2YIQ_MATRIX.transpose()
    yiq_im = np.dot(imRGB.copy(), RGB2YIQ_MATRIX.copy().transpose())
    # imRGB = np.dot(imRGB,RGB2YIQ_MATRIX)
    # plt.imshow(np.dot(imRGB,RGB2YIQ_MATRIX)[:,:,0], cmap="gray")

    # plt.show()
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
    # print (np.histogram(im,256,range = (0,255))[0])
    # print("__________________________________________")

    hist_orig = np.histogram(im,256,range = (0,255))[0]
    # print(hist_orig)
    cumulative_hist = np.cumsum(hist_orig)
    N = cumulative_hist[-1]
    ##(np.asarray(cumulative_hist/N)*255)
    m = cumulative_hist[0]
    # print (m)
    norm_cumulative_hist = np.asarray([((k-m)/(cumulative_hist[-1]-m)*255) for k in cumulative_hist])\
        .astype(np.uint8)
    print(norm_cumulative_hist)
    # print(hist_eq)
    # print("__________________________________________")
    im_eq = norm_cumulative_hist[im]##np.interp(im.flatten(),range(256), norm_cumulative_hist).reshape (im.shape)
    # print (im_eq)
    # print("__________________________________________")
    hist_eq =  np.histogram(im_eq,256,range = (0,255))[0]
    # norm_cumulative_hist = [x*255//N for x in cumulative_hist]
    # norm_cumulative_hist = np.asarray([x*/N for x in cumulative_hist]).astype(np.uint8)
    # print (norm_cumulative_hist)
    # print(norm_cumulative_hist)
    # print(cumulative_hist)
    # print(np.histogram(im,256)[0])
    # print (im)
    # print (hist_orig.shape)
    print ("AAAAAAAAAAAAAAAAAAAAAAAA", dims)
    if (dims>2):
        ##TODO :: CHECK /255 ISSUES (RGB/GREY)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        im_eq /= 255
        img[:, :, 0] = im_eq
        im_eq = yiq2rgb(img)

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
    z = np.asarray([0 for i in range (n_quant+1)])
    q = np.asarray([0 for i in range (n_quant)])
    hist = np.histogram(im,256,range = (0,255))[0]
    cumulative_hist =np.cumsum(hist)
    # print (cumulative_hist)
    max_per_slot = cumulative_hist[-1] // n_quant
    # print ("max_per_slot", max_per_slot)
    z[0] = 0
    i = 0

    ## TODO : loops over 256 are not allowed.
    for c  in range(1,len(cumulative_hist)):
        if (cumulative_hist[c] > max_per_slot*i):
            # print(cumulative_hist[c])
            z[i] = c
            i = i+1
    z[0] = 0
    z [-1] = 255

    print (q)
    print (z,len(z))
    error = np.asarray([0 for i in range(n_iter)])
    cur_iter = 0
    conv = False
    while (cur_iter < n_iter and not conv):
        q = np.asarray([(z[i] * hist[z[i]] + z[i + 1] * hist[z[i + 1]]) // (hist[z[i]] + hist[z[i + 1]])
                        for i in range(len(q))])
        z[1:-1] = np.asarray([(q[i-1]+q[i])//2 for i in range(1, len(q))])
        print ("q__________________")
        print (q)
        print ("z__________________")
        print (z)
        # error[cur_iter] = np.sqrt(np.sum([hist(i)*() for i in hist]))
        cur_iter +=1
    quant_hist =[0]*256
    for i in range(n_quant):
        quant_hist[z[i]:z[i+1]] = [q[i]]*(z[i+1]-z[i])
    quant_hist = np.asarray(quant_hist,np.uint8)
    print (quant_hist)
    print (im)
    im = quant_hist[im]
    print (im)
    print(z.shape, error.shape)
    if (dims>2):
        ##TODO :: CHECK /255 ISSUES (RGB/GREY)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        im = im / 255
        img[:, :, 0] = im
        im_quant = yiq2rgb(img)
    else:
        im_quant = im

    return [im_quant, error]

pic_add = "pulpfiction.jpg"
im = read_image(pic_add, 1)

# x,z,y=histogram_equalize(im)
# print ("______________________________")
# print (z)

plt.imshow(im,cmap = "gray")
plt.show()
im_quant,y = quantize (im,3,10)


# print (im.flatten())
# print(im)
# im_eq,hist_orig,hist_eq= histogram_equalize(im)
# print (hist_eq)
# rgb2yiq(read_image(pic_add, 2))
# print (im.shape)
# imdisplay(pic_add, 1)
# for item in RGB2YIQ_MATRIX:
#     print(item)
# print(yiq2rgb(rgb2yiq(im)).shape)
plt.imshow(im_quant,cmap = "gray")
plt.show()

# print(im.shape[0]*im.shape[1])
# print(im.dtype)
# print (im)
