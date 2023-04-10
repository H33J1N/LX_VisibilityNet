import numpy as np
from PIL import Image
import tensorflow as tf
import cv2


def table_information():
    def generate_form(info):
        text_len = len(info[0])
        print("+", end="")
        for l in range(text_len):
            print("-", end="")
        print("+")

    def generate_info(info):
        print("|", end="")
        print(info[0], end="")
        print("|")

        print("|", end="")
        print(info[1].center(len(info[0])), end="")
        print("|")

    def generate_table(info):
        generate_form(info)
        generate_info(info)
        generate_form(info)

    info = []

    model_name = "Visibility Enhancement Network - Chrominance"
    author_name = "Implemented by Heejin Lee"

    info.append(model_name)
    info.append(author_name)

    generate_table(info)


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def load_images(file, is_resize=False):
    im = cv2.imread(file)
    if is_resize:
        im = cv2.resize(im, (96, 96))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    im = np.array(im, dtype="float32") / 255.0
    y = im[:, :, 0]
    cr = im[:, :, 1] - 0.5
    cb = im[:, :, 2] - 0.5

    y = np.expand_dims(y, -1)
    cr = np.expand_dims(cr, -1)
    cb = np.expand_dims(cb, -1)
    im = np.concatenate([y, cr, cb], -1)

    return im


def load_images_luminance(file, is_resize=False):
    im = Image.open(file).convert('L')
    if is_resize:
        im = im.resize((96, 96))
    im = np.expand_dims(im, -1)

    return np.array(im, dtype="float32") / 255.0


def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    y = cat_image[:, :, 0]
    cr = cat_image[:, :, 1] + 0.5
    cb = cat_image[:, :, 2] + 0.5

    y = np.expand_dims(y, -1)
    cr = np.expand_dims(cr, -1)
    cb = np.expand_dims(cb, -1)

    cat_image = np.concatenate([y, cr, cb], -1)
    im = np.clip(cat_image * 255.0, 0, 255.0)
    im = np.array(im, np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_YCR_CB2BGR)
    cv2.imwrite(filepath, im)


def channel_split(input_tensor):
    hsv = input_tensor
    h = hsv[:, :, :, 0]
    s = hsv[:, :, :, 1]
    v = hsv[:, :, :, 2]

    h = tf.expand_dims(h, -1)
    s = tf.expand_dims(s, -1)
    v = tf.expand_dims(v, -1)

    return h, s, v