from block import *


def visibility_network(input_im):
    with tf.compat.v1.variable_scope('Network'):
        # Luminance Estimation
        down1 = luminance_estimation.conv_down(input_im, 64)
        down2 = luminance_estimation.conv_down(down1, 64)
        down3 = luminance_estimation.conv_down(down2, 128)
        down4 = luminance_estimation.conv_down(down3, 256)
        down5 = luminance_estimation.conv_down(down4, 512)
        down6 = luminance_estimation.conv_down(down5, 1024)

        up6 = luminance_estimation.conv_up(down6, down5, 512)
        up5 = luminance_estimation.conv_up(up6, down4, 256)
        up4 = luminance_estimation.conv_up(up5, down3, 128)
        up3 = luminance_estimation.conv_up(up4, down2, 64)
        up2 = luminance_estimation.conv_up(up3, down1, 64)
        up1 = luminance_estimation.conv_up(up2, input_im, 64, skip_connection=False)

        # Detail Refinement
        conv = tf.concat([up1, input_im], axis=3)
        conv1 = detail_refinement.conv_block(conv, 128)
        conv2 = detail_refinement.conv_block(conv1, 128)
        conv3 = detail_refinement.conv_block(conv2, 128)
        conv4 = detail_refinement.conv_block(conv3, 128)
        conv5 = detail_refinement.conv_block(conv4, 1)

        return conv5
