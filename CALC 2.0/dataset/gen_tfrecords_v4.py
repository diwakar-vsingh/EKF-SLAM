#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function
from coco_classes import coco_classes, calc_classes, calc_class_names
from time import time

import os
import sys
import argparse
import coco
import cv2 as cv
import numpy as np
import tensorflow as tf
vw = 320
vh = 320

# Construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", type=str, metavar='', default="mnt/coco/calc_tfrecords/", help="path to store tfrecords of training input images")
parser.add_argument("-c", "--coco_root", type=str, metavar='', default="mnt/coco", help="path to train image dataset")
parser.add_argument("-n", "--num_files", type=int, metavar='', default=100, help="Number of files to write for train dataset. More files=better randomness")
parser.add_argument("-d", "--debug", default=True, action='store_false', help='Omitting the switch sets args.bool to True, otherwise it is False \
                    {True: Plot training images with mask, False: Convert raw image data and write it in a tfrecord format}')
args = parser.parse_args()
print("Output Dir:\t", args.output_dir, )
print("Coco Dir:\t", args.coco_root)
print("Number of files:", args.num_files)
print("Debug:\t\t", args.debug)

if args.debug:
    print("Importing matplotlib")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    plt.ion()
    imdata = None

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # The following function can be used to convert a value to a type compatible with Tensorflow.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(split, image, mask):
    '''Creates a tf.Example message ready to be written to a file.'''

    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
            'img': _bytes_feature(tf.compat.as_bytes(image.tostring())), 
            'label': _bytes_feature(tf.compat.as_bytes(mask.astype(np.uint8).tostring()))
        }   

    if split == 'val':
        feature = {
            'cl_live': _bytes_feature(tf.compat.as_bytes(image.tostring())),
            'cl_mem': _bytes_feature(tf.compat.as_bytes(mask.tostring()))
        }   
    
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def drawmask(image, mask, nclasses, calc_class_names):
    
    rgb = np.zeros((vh, vw, 3))
    legend = []
    np.random.seed(0)
    for i in range(nclasses):
        c = np.random.rand(3)
        case = mask == i
        if np.any(case):
            legend.append(Patch(facecolor=tuple(c), edgecolor=tuple(c), label=calc_class_names[i]))

            rgb[case, :] = c

    _image = cv.resize(image, (vw, vh)) / 255.0
    _image = 0.3 * _image + 0.7 * rgb

    global imdata
    if imdata is None:
        imdata = plt.imshow(_image)
        f = plt.gca()
        f.axes.get_xaxis().set_ticks([])
        f.axes.get_yaxis().set_ticks([])
    else:
        imdata.set_data(_image)

    lgd = plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.0, 1))

    plt.pause(1e-9)
    plt.draw()
    plt.pause(1)


def tfgenerate():
    # tfrecord for train dataset
    train_writers = []
    for ii in range(args.num_files):
        if args.debug:
            train_writers.append(None)
        else:
            train_writers.append(tf.io.TFRecordWriter(args.output_dir + "train_data%d.tfrecord" % ii)) 

    # tfrecord for val dataset
    val_writer = None if args.debug else tf.io.TFRecordWriter(args.output_dir + "validation_data.tfrecord")


    # Number of coco classes 
    nclasses = len(calc_classes.keys())

    # Initialize class percent for each coco classes
    class_percents = np.zeros((nclasses), dtype=np.float32)

    for split, writer in [('val', val_writer), ('train', train_writers)]:
        
        # Load coco dataset
        dataset = coco.CocoDataset()
        dataset.load_coco(args.coco_root, split)

        # Must call before using the dataset
        dataset.prepare()

        # Print image counts and number of classes
        print("Image Count: {}".format(len(dataset.image_ids)))
        print("COCO Class Count: {}".format(dataset.num_classes))
        print("CALC Class Count: {}".format(nclasses))

        count = 1
        print(dataset.image_ids)
        for image_id in dataset.image_ids:
            print("Working on sample %d" % image_id)
            if split == 'val':
                cl_live = cv.cvtColor(cv.resize(
                    cv.imread("CampusLoopDataset/live/Image%s.jpg" % (str(count).zfill(3))),
                    (vw, vh), interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
                
                cl_mem = cv.cvtColor(cv.resize(
                        cv.imread("CampusLoopDataset/memory/Image%s.jpg" % (str(count).zfill(3))),
                        (vw, vh), interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)

            image = cv.resize(dataset.load_image(image_id), (vw, vh), interpolation=cv.INTER_AREA)
            masks, class_ids = dataset.load_mask(image_id)
            mask_label = np.zeros((vh, vw, nclasses), dtype=np.bool)
            for i in range(masks.shape[2]):
                cid = calc_classes[coco_classes[class_ids[i]][1]]

                mask_label[:, :, cid] = np.logical_or(mask_label[:, :, cid], 
                    cv.resize(masks[:, :, i].astype(np.uint8), (vw, vh), 
                    interpolation=cv.INTER_NEAREST).astype(np.bool))

            # No labels for BG. Make them!
            mask_label[:, :, 0] = np.logical_not(np.any(mask_label[:, :, 1:], axis=2))

            if split == 'train':
                cp = np.mean(mask_label, axis=(0, 1))
                class_percents += (1.0 / count) * (cp - class_percents)
            mask = np.argmax(mask_label, axis=-1)

            if args.debug:
                # Showing images with mask
                drawmask(image, mask, nclasses, calc_class_names)
            else:
                if split == 'val':
                    # Writing a TFRecord file
                    val_writer.write(serialize_example(split, cl_live, cl_mem))
                else:
                    # Writing a TFRecord file
                    train_writers[np.random.randint(0, args.num_files)].write(serialize_example(split, image, mask))

            if split == 'val' and image_id == 99:
                break
            count += 1

            
    class_weights = 1.0 / class_percents
    with open('loss_weights.txt', 'w') as f:
        s = ''
        for w in class_weights:
            s += str(w) + ' '
        f.write(s)


if __name__ == "__main__":
    tfgenerate()