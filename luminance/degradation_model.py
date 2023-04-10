import tensorflow as tf


def degradation_ratio(condition):
    if condition == 'sunlight':
        return 0.1
    elif condition == 'overcast':
        return 0.02
    elif condition == 'office':
        return 0.005
    else:
        print("[!] Invalid condition")
        exit(0)


def grayscale_to_lum(image, C_r, L_w):
    gamma = 2.2
    luminance = ((image / 255) ** gamma) * (L_w - (L_w / C_r)) + L_w / C_r
    return luminance


def degradation(source, lumin_ratio):
    im_ori = source
    im_ori = im_ori * 255.0
    im_ori = tf.clip_by_value(im_ori, clip_value_min=0, clip_value_max=255.0)

    im_gray = im_ori

    """
    [Degradation equation]
    D: source image under normal light
    S: source image under bright light
    """

    L_ws = 400  # maximum luminance at S
    L_wd = 400
    C_rs = 150  # contrast at S
    C_rd = 150

    # ratio_lum = L_ws / L_wd
    L_e = lumin_ratio * L_ws  # L_e = L_r(reflection from display) + L_v(lumination at light)

    gamma = 2.2

    # contrast perceived from light difference
    C_rs_p = (L_ws + L_e) / ((L_ws / C_rs) + L_e)
    # Maximum used Luminance
    L_wd_p = L_ws * (C_rs_p / C_rs)
    # Image gray scale according to reduced contrast
    G_wd_p = (((L_wd_p - (L_wd / C_rd)) / (L_wd - (L_wd / C_rd))) ** (1 / gamma)) * 256

    L_ws_img = grayscale_to_lum(im_gray, C_rs, L_ws)
    max_val = tf.reduce_max(L_ws_img) + L_e

    G_p = ((L_ws_img + L_e) / (max_val + 1e-5)) ** (1 / gamma)  #
    G_p_a = G_p
    L_g_p = ((G_p_a) ** gamma) * L_ws
    output = ((L_g_p - (L_wd / C_rd)) / (L_ws - (L_ws / C_rs)))
    output = ((output) ** (1 / gamma)) * L_wd_p

    output_max = tf.reduce_max(output)
    output_min = tf.reduce_min(output)

    output = (output - output_min) * G_wd_p / ((output_max - output_min) + 1e-9)
    ratio = output / (im_gray + 0.0000000000005)
    data = im_ori * ratio
    data = data / 255.0

    return data
