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


def variance(img1, img2, size=11, sigma=1.5):
    window = ssim_equation(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a different scale)
    C = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    value = (2 * sigma12 + C) / (sigma1_sq + sigma2_sq + C)
    value = tf.reduce_mean(value)

    return value


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


def adaptive_mask(y_true, y_pred):
    mask_list = []

    for b in range(8):
        y_true_copy = tf.identity(y_true)
        y_true_batch = tf.identity(y_true_copy[b, :, :, :])
        y_true_mask = tf.identity(y_true_batch)

        y_pred_copy = tf.identity(y_pred)
        y_pred_batch = tf.identity(y_pred_copy[b, :, :, :])
        y_pred_mask = tf.identity(y_pred_batch)

        mask_1 = tf.ones_like(y_true_mask)
        mask_0 = tf.zeros_like(y_true_mask)

        deg_max = tf.reduce_max(y_pred_mask)
        deg_mean = tf.reduce_mean(y_pred_mask)
        src_mean = tf.reduce_mean(y_true_mask)

        max_value = tf.math.maximum(deg_max, src_mean)

        mask_dark = tf.where(y_true_mask < src_mean, mask_1, mask_0)
        mask_bright = tf.where(y_true_mask > max_value, mask_1, mask_0)

        mask = tf.cond(src_mean > deg_max, lambda: mask_bright, lambda: mask_dark)
        mask_list.append(mask)

    mask_tensor = tf.stack(mask_list)

    gt_masked = tf.multiply(mask_tensor, y_true)
    deg_masked = tf.multiply(mask_tensor, y_pred)

    return gt_masked, deg_masked


def var_loss(y_true, y_pred):
    variance_loss = variance(tf.expand_dims(y_pred[:, :, :, 0], -1), tf.expand_dims(y_true[:, :, :, 0], -1))
    return variance_loss


def ssim_loss(y_true, y_pred):
    SSIM_loss = ssim(tf.expand_dims(y_pred[:, :, :, 0], -1), tf.expand_dims(y_true[:, :, :, 0], -1))
    return SSIM_loss



