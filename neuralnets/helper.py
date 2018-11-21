import numpy as np
import scipy.misc
import scipy.io



def imread(path):
    """
    read image
    """
    return scipy.misc.imread(path).astype('float32')   # returns RGB format

def imsave(path, img):
    """
    save image
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    return scipy.misc.imsave(path, img)

def imgpreprocess(image, mean_pixel):
    """
    minus mean pixel for normalization
    """
    image = np.ndarray.reshape(image,((1,) + image.shape)) 
    return (image - mean_pixel).astype('float32')

def imgunprocess(image, mean_pixel):
    """
    add mean pixel to recovery image
    """
    return (image + mean_pixel).astype('float32')