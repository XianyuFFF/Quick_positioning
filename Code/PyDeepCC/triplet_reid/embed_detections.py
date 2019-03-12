#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import os

import h5py
import json
import numpy as np
import tensorflow as tf

from .aggregators import AGGREGATORS
from .common import *
from .duke_utils import *
import scipy.io as sio
import functools


def flip_augment(image, fid, pid):
    """ Returns both the original and the horizontal flip of an image. """
    images = tf.stack([image, tf.reverse(image, [1])])
    return images, tf.stack([fid]*2), tf.stack([pid]*2)


def five_crops(image, crop_size):
    """ Returns the central and four corner crops of `crop_size` from `image`. """
    image_size = tf.shape(image)[:2]
    crop_margin = tf.subtract(image_size, crop_size)
    assert_size = tf.assert_non_negative(
        crop_margin, message='Crop size must be smaller or equal to the image size.')
    with tf.control_dependencies([assert_size]):
        top_left = tf.floor_div(crop_margin, 2)
        bottom_right = tf.add(top_left, crop_size)
    center       = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    top_left     = image[:-crop_margin[0], :-crop_margin[1]]
    top_right    = image[:-crop_margin[0], crop_margin[1]:]
    bottom_left  = image[crop_margin[0]:, :-crop_margin[1]]
    bottom_right = image[crop_margin[0]:, crop_margin[1]:]
    return center, top_left, top_right, bottom_left, bottom_right


def embed_detections(experiment_root, detection_generator, num_detections, file_name, check_point=None, loading_threads=8,
                     batch_size=256, flip_augment=False, crop_augment=False, aggregator=None, quiet=False):

    tf.reset_default_graph()
    # Load the args from the original experiment.
    args_file = os.path.join(experiment_root, 'args.json')

    if os.path.isfile(args_file):
        if not quiet:
            print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)

        # # Add arguments from training.
        # for key, value in args_resumed.items():
        #     args.__dict__.setdefault(key, value)

        # A couple special-cases and sanity checks
        if (args_resumed['crop_augment']) == (crop_augment is None):
            print('WARNING: crop augmentation differs between training and '
                  'evaluation.')
        image_root = args_resumed['image_root']
    else:
        raise IOError('`args.json` could not be found in: {}'.format(args_file))

    # Check a proper aggregator is provided if augmentation is used.
    if flip_augment or crop_augment == 'five':
        if aggregator is None:
            print('ERROR: Test time augmentation is performed but no aggregator'
                  'was specified.')
            exit(1)
    else:
        if aggregator is not None:
            print('ERROR: No test time augmentation that needs aggregating is '
                  'performed but an aggregator was specified.')
            exit(1)

    if not quiet:
        print('Evaluating using the following parameters:')

    # Load the data from the CSV file.

    net_input_size = (args_resumed['net_input_height'], args_resumed['net_input_width'])
    pre_crop_size = (args_resumed['pre_crop_height'], args_resumed['pre_crop_width'])

    # Setup a tf Dataset generator

    dataset = tf.data.Dataset.from_generator(detection_generator, tf.float32,
                                             tf.TensorShape([net_input_size[0], net_input_size[1], 3]))

    modifiers = ['original']
    if flip_augment:
        dataset = dataset.map(flip_augment)
        dataset = dataset.apply(tf.contrib.data.unbatch())
        modifiers = [o + m for m in ['', '_flip'] for o in modifiers]

    if crop_augment == 'center':
        dataset = dataset.map(lambda im, fid, pid:
                              (five_crops(im, net_input_size)[0], fid, pid))
        modifiers = [o + '_center' for o in modifiers]
    elif crop_augment == 'five':
        dataset = dataset.map(lambda im, fid, pid:
                              (tf.stack(five_crops(im, net_input_size)), [fid] * 5, [pid] * 5))
        dataset = dataset.apply(tf.contrib.data.unbatch())
        modifiers = [o + m for o in modifiers for m in [
            '_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]
    elif crop_augment == 'avgpool':
        modifiers = [o + '_avgpool' for o in modifiers]
    else:
        modifiers = [o + '_resize' for o in modifiers]

    # Group it back into PK batches.
    dataset = dataset.batch(batch_size)

    # Overlap producing and consuming.
    dataset = dataset.prefetch(batch_size)
    images = dataset.make_one_shot_iterator().get_next()

    # Create the model and an embedding head.
    model = import_module('.nets.' + args_resumed['model_name'], 'triplet_reid')
    head = import_module('.heads.' + args_resumed['head_name'], 'triplet_reid')

    endpoints, body_prefix = model.endpoints(images, is_training=False)
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args_resumed['embedding_dim'], is_training=False)

    with h5py.File(file_name, 'w') as f_out, tf.Session() as sess:
        # Initialize the network/load the checkpoint.
        if check_point is None:
            check_point = tf.train.latest_checkpoint(experiment_root)
        else:
            check_point = os.path.join(experiment_root, check_point)
        if not quiet:
            print('Restoring from checkpoint: {}'.format(check_point))
        tf.train.Saver().restore(sess, check_point)

        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = np.zeros(
            (num_detections * len(modifiers), args_resumed['embedding_dim']), np.float32)

        print(emb_storage.shape)

        for start_idx in count(step=batch_size):
            try:
                emb = sess.run(endpoints['emb'])
                print('\rEmbedded batch {}-{}/{}'.format(
                    start_idx, start_idx + len(emb), len(emb_storage)),
                    flush=True, end='')
                emb_storage[start_idx:start_idx + len(emb)] = emb
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.

        if not quiet:
            print("Done with embedding, aggregating augmentations...", flush=True)

        emb_dataset = f_out.create_dataset('emb', data=emb_storage)
        # Store information about the produced augmentation and in case no crop
        # augmentation was used, if the images are resized or avg pooled.
        f_out.create_dataset('augmentation_types', data=np.asarray(modifiers, dtype='|S'))
