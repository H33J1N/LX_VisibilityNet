import tensorflow as tf


class luminance_estimation:
    def conv_down(conv, channel):
        down = tf.compat.v1.layers.conv2d(conv, channel, 3, 2, padding='same')
        down = tf.compat.v1.layers.batch_normalization(down)
        down = tf.nn.relu(down)

        return down

    def conv_up(down_conv, up_conv, channel, skip_connection=True):
        up = tf.compat.v1.image.resize_nearest_neighbor(down_conv, (tf.shape(up_conv)[1], tf.shape(up_conv)[2]))
        up = tf.compat.v1.layers.conv2d(up, channel, 3, 1, padding='same')
        up = tf.compat.v1.layers.batch_normalization(up)
        up = tf.nn.relu(up)

        if skip_connection:
            up = up + up_conv

        return up


class detail_refinement:
    def conv_block(conv, channel, is_last=False):
        if is_last ==  False:
            block = tf.compat.v1.layers.conv2d(conv, channel, 3, 1, padding='same', activation=tf.nn.relu)
        if is_last:
            block = tf.compat.v1.layers.conv2d(conv, channel, 3, 1, padding='same')
        return block
