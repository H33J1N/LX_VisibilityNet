import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import *
from tensorboardX import SummaryWriter
import math


def ssim_equation(size, sigma):
    """
    Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = ssim_equation(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    value = (((2 * mu1_mu2 + C1) ** 1) * ((2 * sigma12 + C2) ** 1)) / (((mu1_sq + mu2_sq + C1) ** 1) *
                                                                       ((sigma1_sq + sigma2_sq + C2) ** 1))
    value = tf.reduce_mean(value)

    return value


def ssim_loss(y_true, y_pred):
    SSIM_loss = ssim(tf.expand_dims(y_pred[:, :, :, 0], -1), tf.expand_dims(y_true[:, :, :, 0], -1))
    return SSIM_loss


def ssim_chrominance(src_u, src_v, deg_u, deg_v):
    src_u_p = tf.where(src_u > 0, src_u, src_u * 0)
    src_u_n = tf.where(src_u < 0, src_u, src_u * 0)

    src_v_p = tf.where(src_v > 0, src_v, src_v * 0)
    src_v_n = tf.where(src_v < 0, src_v, src_v * 0)

    deg_u_p = tf.where(src_u > 0, deg_u, deg_u * 0)
    deg_u_n = tf.where(src_u < 0, deg_u, deg_u * 0)

    deg_v_p = tf.where(src_v > 0, deg_v, deg_v * 0)
    deg_v_n = tf.where(src_v < 0, deg_v, deg_v * 0)

    ssim_u = 2 - (ssim_loss(src_u_p, deg_u_p) + ssim_loss((-1) * src_u_n, (-1) * deg_u_n))
    ssim_v = 2 - (ssim_loss(src_v_p, deg_v_p) + ssim_loss((-1) * src_v_n, (-1) * deg_v_n))

    ssim_value = ssim_u + ssim_v

    return ssim_value


def cosine_loss(src_u, src_v, deg_u, deg_v):
    src_c = tf.concat([src_u, src_v], -1)
    deg_c = tf.concat([deg_u, deg_v], -1)

    src_l2_norm = tf.math.l2_normalize(src_c, axis=-1)
    deg_l2_norm = tf.math.l2_normalize(deg_c, axis=-1)

    cosine_c = tf.multiply(src_l2_norm, deg_l2_norm)
    cosine = tf.reduce_sum(cosine_c, -1)
    cosine = tf.reduce_mean(cosine)
    cos_value = 0.5 * (1 - cosine)

    return cos_value
