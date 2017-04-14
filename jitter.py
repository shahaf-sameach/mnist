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
        temp = rotate(temp, angle = np.random.randint(-20,21,1), reshape=False)
    # if np.random.random() > prob:
    #     temp = elastic_transform(temp)
    return temp


def elastic_transform(image, kernel_dim=13, sigma=6, alpha=36, negated=False):
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

    return result

def create_2d_gaussian(dim, sigma):
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

    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_test = X_test.astype('float32')
    X_test /= 255

    plt.figure()
    f, ax = plt.subplots(2, 2)
    ax[0,0].imshow(X_test[1,0,:,:])
    ax[0,1].imshow(elastic_transform(X_test[1,0,:,:]))
    plt.show()




