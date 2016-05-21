import numpy as np
import math
from scipy.ndimage.interpolation import rotate, shift
from scipy.signal import convolve2d
from numpy.random import random_integers


def rand_jitter(temp, prob=0.5):
    np.random.seed(1337)  # for reproducibility

    if np.random.random() > prob:
        temp[np.random.randint(0,28,1), :] = 0
    if np.random.random() > prob:
        temp[:, np.random.randint(0,28,1)] = 0
    if np.random.random() > prob:
        temp = shift(temp, shift=(np.random.randint(-3,4,2)))
    if np.random.random() > prob:
        temp = rotate(temp, angle = np.random.randint(-15,16,1), reshape=False)
    if np.random.random() > prob:
        temp = elastic_transform(temp)
    return temp

def add_h_noise(data):
  x_data = data[:,1:]
  y_data = data[:,0]
  x_data = x_data.reshape(x_data.shape[0], 1, 28, 28)
  x_data = x_data.astype('float32')
  for i in range(0,x_data.shape[0]):
    x_data[np.random.randint(0,28,1), :] = 0
    x_data[j,0, :, :] = h_noise(x_data[j,0,:,:])


def h_noise(temp):
  temp[np.random.randint(0,28,1), :] = 0

def v_noise(temp):
  temp[np.random.randint(0,28,1), :] = 0

def shift_noise(temp):
  data = np.copy(temp)
  for i in range(-3,4):
    for j in range(-3,4):
      temp = shift(temp, shift=(np.random.randint(-3,4,2)))
      data = np.append(data, temp)


def elastic_transform(image, kernel_dim=13, sigma=6, alpha=36, negated=False):
    """
    This method performs elastic transformations on an image by convolving 
    with a gaussian kernel.
    NOTE: Image dimensions should be a sqaure image
    
    :param image: the input image
    :type image: a np nd array
    :param kernel_dim: dimension(1-D) of the gaussian kernel
    :type kernel_dim: int
    :param sigma: standard deviation of the kernel
    :type sigma: float
    :param alpha: a multiplicative factor for image after convolution
    :type alpha: float
    :param negated: a flag indicating whether the image is negated or not
    :type negated: boolean
    :returns: a nd array transformed image
    """
    
    # convert the image to single channel if it is multi channel one
    # if image.ndim == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check if the image is a negated one
    # if not negated:
    #     image = 255-image

    # check if the image is a square one
    # if image.shape[0] != image.shape[1]:
    #     raise ValueError("Image should be of sqaure form")

    # check if kernel dimesnion is odd
    # if kernel_dim % 2 == 0:
    #     raise ValueError("Kernel dimension should be odd")

    # create an empty image
    result = np.zeros(image.shape)

    # create random displacement fields
    displacement_field_x = np.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha
    displacement_field_y = np.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha

    # create the gaussian kernel
    kernel = create_2d_gaussian(kernel_dim, sigma)

    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel)
    displacement_field_y = convolve2d(displacement_field_y, kernel)

    # make the distortrd image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields
    
    for row in xrange(image.shape[1]):
        for col in xrange(image.shape[0]):
            low_ii = row + int(math.floor(displacement_field_x[row, col]))
            high_ii = row + int(math.ceil(displacement_field_x[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
               or high_jj >= image.shape[0] - 1:
                continue

            res = image[low_ii, low_jj]/4 + image[low_ii, high_jj]/4 + \
                    image[high_ii, low_jj]/4 + image[high_ii, high_jj]/4

            result[row, col] = res
    
    # if the input image was not negated, make the output image also a non 
    # negated one
    # if not negated:
    #     result = 255-result

    return result

def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma
    
    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int
    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float
    
    :returns: a np 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = np.zeros((dim, dim), dtype=np.float16)

    # calculate the center point
    center = dim/2

    # calculate the variance
    variance = sigma ** 2
    
    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2*variance
            
            kernel[x,y] = coeff * np.exp(-1. * numerator/denom)
    
    # normalise it
    return kernel/sum(sum(kernel))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from keras.datasets import mnist


    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    plt.figure()
    f, ax = plt.subplots(2, 2)
    ax[0,0].imshow(X_test[1,0,:,:])
    ax[0,1].imshow(elastic_transform(X_test[1,0,:,:]))
    plt.show()




