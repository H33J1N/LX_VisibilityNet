from __future__ import print_function

import os
import time
import random

from network import visibility_network
from degradation_model import *
from loss import *
from tqdm import tqdm


class visibility_enhance(object):
    def __init__(self, sess, lr, condition):
        self.sess = sess
        self.condition = condition
        self.base_lr = lr

        self.input_source = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='input_source')
        self.output = visibility_network(self.input_source)

        y_true = tf.clip_by_value(self.input_source, clip_value_min=0, clip_value_max=1)
        self.output = tf.clip_by_value(self.output, clip_value_min=0, clip_value_max=1)

        deg_y_pred = degradation(self.output, degradation_ratio(condition))

        y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=1)
        deg_y_pred = tf.clip_by_value(deg_y_pred, clip_value_min=0, clip_value_max=1)

        self.y_true_adaptive, self.deg_y_pred_adaptive = adaptive_mask(y_true, deg_y_pred)

        self.variation_loss = 1 - var_loss(y_true, deg_y_pred)
        self.adaptive_loss = 1 - ssim_loss(self.y_true_adaptive, self.deg_y_pred_adaptive)

        self.loss = 0.25 * self.variation_loss + 1 * self.adaptive_loss

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.compat.v1.train.exponential_decay(self.base_lr, self.global_step, 50, 0.8)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        # To set the training stable
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1, 1) + var * 1e-4, var) for grad, var in grads_and_vars]

        self.train_op = optimizer.apply_gradients(capped_gvs)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        print("[*] Initialize model successfully...")

    def train(self, train_data, valid_data, batch_size, patch_size, epoch, sample_dir, ckpt_dir, eval_every_epoch):
        numBatch = len(train_data) // int(batch_size)

        load_model_status, global_step = self.load(self.saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training with start epoch %d start iter %d : " % (start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            t = tqdm(range(start_step, numBatch))
            for batch_id in t:
                # generate data for a batch
                batch_input = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_data[image_id].shape
                    x = random.randint(0, int(0.5 * (h - patch_size)))
                    y = random.randint(0, int(0.5 * (w - patch_size)))

                    rand_mode = random.randint(0, 7)
                    batch_input[patch_id, :, :, :] = data_augmentation(
                        train_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)

                    image_id = (image_id + 1) % len(train_data)
                    if image_id == 0:
                        tmp = list(train_data)
                        random.shuffle(list(tmp))
                        train_data = tmp

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input_source: batch_input})
                t.set_description("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"  % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Validation
            if (epoch + 1) % eval_every_epoch == 0:
                self.validation(epoch + 1, valid_data, sample_dir=sample_dir)
                self.save(self.saver, iter_num, ckpt_dir, self.condition)

        print("[*] Finish training")

    def validation(self, epoch_num, valid_data, sample_dir):
        print("[*] Evaluating for epoch %d..." % epoch_num)
        t = tqdm(range(len(valid_data)))
        for idx in t:
            input_eval = np.expand_dims(valid_data[idx], axis=0)
            result = self.sess.run(self.output, feed_dict={self.input_source: input_eval})
            save_images(os.path.join(sample_dir, 'eval_%d_%d.png' % (idx + 1, epoch_num)), input_eval, result)
            t.set_description("Eval: [%2d] [%4d/%4d]" % (epoch_num, idx + 1, len(valid_data)))

    def save(self, saver, iter_num, ckpt_dir, condition):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % condition)
        saver.save(self.sess, os.path.join(ckpt_dir, condition), global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_data, test_data_names, ckpt_dir, save_dir):
        tf.compat.v1.global_variables_initializer().run()

        load_model_status, _ = self.load(self.saver, ckpt_dir)
        print("[*] Reading checkpoint...")
        if load_model_status:
            print("[*] Load weights successfully...")

        print("[*] Testing...")

        for idx in range(len(test_data)):
            [_, name] = os.path.split(test_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_test = np.expand_dims(test_data[idx], axis=0)
            result = self.sess.run(self.output, feed_dict={self.input_source: input_test})
            save_images(os.path.join(save_dir, name + "." + suffix), result)
            print("Saving:", save_dir + name + "." + suffix)
