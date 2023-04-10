"""
Visibility Enhancement Network: Luminance
@Author: Heejin Lee
"""

from __future__ import print_function
import os
import argparse
from glob import glob

import tensorflow as tf
from model import *
from utils import *
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0, 1", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.8, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--condition', dest='condition', type=str, default='sunlight', help='office , overcast, sunlight')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='number of total epoches')
parser.add_argument('--lr', dest='lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=96, help='patch size')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=1,
                    help='evaluating and saving checkpoints every # epoch')

parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpt/', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./valid/', help='directory for evaluating outputs')
parser.add_argument('--save_dir', dest='save_dir', default='./result/', help='directory for testing outputs')

parser.add_argument('--train_dir', dest='train_dir', default='../dataset/train/', help='directory for training inputs')
parser.add_argument('--valid_dir', dest='valid_dir', default='../dataset/valid/', help='directory for validate inputs')
parser.add_argument('--test_dir', dest='test_dir', default='../dataset/test/', help='directory for testing inputs')

args = parser.parse_args()

# Set the directory to automatically saving according to condition
args.ckpt_dir = args.ckpt_dir + "{}/".format(args.condition)
args.sample_dir = args.sample_dir + "{}/".format(args.condition)
args.save_dir = args.save_dir + "{}/".format(args.condition)


def visibility_train(visibility_enhance):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    train_data = []

    train_data_names = glob('{}/*.png'.format(args.train_dir))
    train_data_names.sort()
    print('[*] Number of training data: %d' % len(train_data_names))

    for idx in range(len(train_data_names)):
        train_im = load_images(train_data_names[idx], is_resize=True)
        train_data.append(train_im)

    valid_data = []

    valid_data_name = glob('{}/*.*'.format(args.valid_dir))

    for idx in range(len(valid_data_name)):
        valid_im = load_images(valid_data_name[idx])
        valid_data.append(valid_im)

    visibility_enhance.train(train_data, valid_data, batch_size=args.batch_size,
                             patch_size=args.patch_size, epoch=args.epoch, sample_dir=args.sample_dir,
                             ckpt_dir=args.ckpt_dir, eval_every_epoch=args.eval_every_epoch)


def visibility_test(visibility_enhance):
    if args.test_dir is None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_data_name = glob(os.path.join(args.test_dir) + '/*.*')
    test_data = []
    for idx in range(len(test_data_name)):
        test_im = load_images(test_data_name[idx])
        test_data.append(test_im)

    visibility_enhance.test(test_data, test_data_name, ckpt_dir=args.ckpt_dir, save_dir=args.save_dir)


def main(_):
    table_information()
    if args.use_gpu:
        print("[*] GPU MODE")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        tf.reset_default_graph()
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            model = visibility_enhance(sess, args.lr, args.condition)
            if args.phase == 'train':
                visibility_train(model)
            elif args.phase == 'test':
                visibility_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU MODE")
        with tf.compat.v1.Session() as sess:
            model = visibility_enhance(sess, args.lr, args.condition)
            if args.phase == 'train':
                visibility_train(model)
            elif args.phase == 'test':
                visibility_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()
