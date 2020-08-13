#!/usr/bin/env python3

import os
import sys
import datetime
import numpy as np
import utils_v2
import layers_v2
import tensorflow as tf
import tf_slim as slim
# from tensorflow.contrib import slim
from time import time
from dataset.coco_classes import calc_classes
from multiprocessing import cpu_count as n_cpus
from dataset.gen_tfrecords import vw as __vw
from dataset.gen_tfrecords import vh as __vh

N_CLASSES = len(calc_classes.keys())
vw = 256
vh = 192  # Need 128 since we go down by factors of 2

with open('dataset/loss_weights.txt', 'r') as f:
    _weights = np.reshape(np.fromstring(f.read(),
                                        sep=' ', dtype=np.float32,
                                        count=N_CLASSES), (1, 1, 1, -1))

FLAGS = tf.compat.v1.flags.FLAGS
if __name__ == '__main__':
    tf.compat.v1.flags.DEFINE_string("mode", "train", "train, pr, ex, or best")
    tf.compat.v1.flags.DEFINE_string("model_dir", "model", "Estimator model_dir")
    tf.compat.v1.flags.DEFINE_string("data_dir", "dataset/CampusLoopDataset", "Path to data")
    tf.compat.v1.flags.DEFINE_string("title", "Precision-Recall Curve", "Plot title")
    tf.compat.v1.flags.DEFINE_integer("n_include", 5, "")
    tf.compat.v1.flags.DEFINE_integer("steps", 200000, "Training steps")
    tf.compat.v1.flags.DEFINE_string(
        "hparams", "",
        "A comma-separated list of `name=value` hyperparameter values. This flag "
        "is used to override hyperparameter settings when manually "
        "selecting hyperparameters.")
    tf.compat.v1.flags.DEFINE_integer("batch_size", 12, "Size of mini-batch.")
    tf.compat.v1.flags.DEFINE_string("netvlad_feat", None,
                               "Binary base file for NetVLAD features. If you did this for dataset XX, "
                               "the program will look for XX_db.bin and XX_q.bin")
    tf.compat.v1.flags.DEFINE_string("input_dir", "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/coco/calc_tfrecords/",
                               "tfrecords dir")
    tf.compat.v1.flags.DEFINE_boolean("include_calc", False, "Include original calc in pr curve"
                                                       "Place in 'calc_model' directory if this is set")
    tf.compat.v1.flags.DEFINE_string("image_fl", "", "")


def create_input_fn(split, batch_size):
    """Returns input_fn for tf.estimator.Estimator.
    tf.estimator.Estimator: Estimator class to train and evaluate TensorFlow models.
    Reads tfrecord file and constructs input_fn for training

    Args:
    tfrecord: the .tfrecord file
    batch_size: The batch size!

    Returns:
    input_fn for tf.estimator.Estimator.

    Raises:
    IOError: If test.txt or dev.txt are not found.
    """

    def input_fn():
        """input_fn for tf.estimator.Estimator."""

        indir = FLAGS.input_dir
        tfrecord = 'train_data*.tfrecord' if split == 'train' else 'validation_data.tfrecord'

        def parser(serialized_example):

            features_ = {'img': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.string)}

            if split != 'train':
                features_['cl_live'] = tf.io.FixedLenFeature([], tf.string)
                features_['cl_mem'] = tf.io.FixedLenFeature([], tf.string)

            fs = tf.io.parse_single_example(
                serialized=serialized_example,
                features=features_
            )

            fs['img'] = tf.reshape(tf.cast(tf.io.decode_raw(fs['img'], tf.uint8),
                                           tf.float32) / 255.0, [__vh, __vw, 3])
            fs['label'] = tf.reshape(tf.io.decode_raw(fs['label'], tf.uint8), [__vh, __vw])
            fs['label'] = tf.cast(tf.one_hot(fs['label'], N_CLASSES), tf.float32)

            if split != 'train':
                fs['cl_live'] = tf.reshape(tf.cast(tf.io.decode_raw(fs['cl_live'], tf.uint8),
                                                   tf.float32) / 255.0, [__vh, __vw, 3])
                fs['cl_mem'] = tf.reshape(tf.cast(tf.io.decode_raw(fs['cl_mem'], tf.uint8),
                                                  tf.float32) / 255.0, [__vh, __vw, 3])
                fs['cl_live'] = tf.reshape(tf.image.resize(fs['cl_live'],
                                                                  (vh, vw)), [vh, vw, 3])
                fs['cl_mem'] = tf.reshape(tf.image.resize(fs['cl_mem'],
                                                                 (vh, vw)), [vh, vw, 3])

            return fs

        if split == 'train':
            files = tf.data.Dataset.list_files(indir + tfrecord, shuffle=True,
                                               seed=np.int64(time()))
        else:
            files = [indir + tfrecord]

        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.shuffle(400, seed=np.int64(time()))
        dataset = dataset.repeat()
        # dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(400, seed=np.int64(time())))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(parser,
                                                                   batch_size if split == 'train' else batch_size // 3,
                                                                   num_parallel_calls=n_cpus() // 2))
        dataset = dataset.prefetch(buffer_size=2)

        return dataset

    return input_fn


def vss(images, is_training=False, ret_descr=False, reuse=False,
        ret_c_centers=False, ret_mu=False, ret_c5=False):
    # Variational Semantic Segmentator
    with tf.compat.v1.variable_scope("VSS", reuse=reuse):
        images = tf.identity(images, name='images')
        batch_norm_params = {
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'fused': True,  # Use fused batch norm if possible.
        }

        with slim.arg_scope(
                [slim.conv2d],
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                activation_fn=lambda x: tf.nn.elu(x),
                weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                	scale=1.0, mode="fan_avg", distribution=("uniform" if False else "truncated_normal")),
                padding='SAME'):

            ### Encoder ####################################
            r1 = slim.conv2d(images, 32, [3, 3])

            r2 = slim.conv2d(r1, 16, [1, 1])
            r3 = slim.conv2d(r2, 32, [3, 3]) + r1

            r4 = slim.conv2d(r3, 16, [1, 1])
            r5 = slim.conv2d(r4, 32, [3, 3]) + r3

            p1 = tf.compat.v1.layers.max_pooling2d(r5, [2, 2], 2, padding='same')

            d21 = slim.conv2d(p1, 64, [3, 3])
            d22 = slim.conv2d(d21, 64, [3, 3])
            p2 = tf.compat.v1.layers.max_pooling2d(d22, [2, 2], 2, padding='same')

            d31 = slim.conv2d(p2, 128, [3, 3])
            d32 = slim.conv2d(d31, 128, [3, 3])
            p3 = tf.compat.v1.layers.max_pooling2d(d32, [2, 2], 2, padding='same')

            d41 = slim.conv2d(p3, 256, [3, 3])
            d42 = slim.conv2d(d41, 256, [3, 3])
            p4 = tf.compat.v1.layers.max_pooling2d(d42, [2, 2], 2, padding='same')

            d51 = slim.conv2d(p4, 512, [3, 3])
            d52 = slim.conv2d(d51, 512, [3, 3])

            #### Latent vars #######################################

            # Dont slice since we dont want to compute twice as many feature maps for nothing
            mu = slim.conv2d(d52, 4 * (1 + N_CLASSES), [3, 3], scope="mu",
                             activation_fn=None,
                             normalizer_fn=None,
                             normalizer_params=None
                             )

            if ret_mu:
                return mu

            sh = mu.get_shape().as_list()
            c_centers = tf.compat.v1.get_variable('offset',
                                        initializer=tf.random.normal([1, sh[1], sh[2], sh[3]]),
                                        trainable=True)

            res = mu - c_centers

            # Intra normalization and overall normalization
            l2 = tf.math.l2_normalize
            descr = l2(tf.reshape(l2(res, axis=-1), [-1, sh[3] * sh[1] * sh[2]]),
                       axis=-1, name='descriptor')
            if ret_c5:
                return descr, r5
            if ret_c_centers:
                return descr, c_centers
            if ret_descr:
                return descr

            log_sig_sq = slim.conv2d(d52, 4 * (1 + N_CLASSES), [3, 3], scope="log_sig_sq",
                                     activation_fn=None,
                                     normalizer_fn=None,
                                     normalizer_params=None
                                     )

            # z = mu + sigma * epsilon
            # epsilon is a sample from a N(0, 1) distribution
            eps = tf.random.normal(tf.shape(input=mu), 0.0, 1.0, dtype=tf.float32)

            # Random normal variable for decoder :D
            z = mu + tf.sqrt(tf.exp(log_sig_sq)) * eps

            ### Decoder ####################################
            decoders = []
            for i in range(1 + N_CLASSES):
                u41 = tf.compat.v1.depth_to_space(input=slim.conv2d(z[:, :, :, i:(i + 4)], 128, [3, 3]), block_size=2)
                u42 = slim.conv2d(u41, 128, [3, 3])
                u43 = slim.conv2d(u42, 128, [3, 3])

                u31 = slim.conv2d(tf.compat.v1.depth_to_space(input=u43, block_size=2), 64, [3, 3])
                u32 = slim.conv2d(u31, 64, [3, 3])
                u33 = slim.conv2d(u32, 64, [3, 3])

                u21 = slim.conv2d(tf.compat.v1.depth_to_space(input=u33, block_size=2), 32, [3, 3])
                u22 = slim.conv2d(u21, 32, [3, 3])
                u23 = slim.conv2d(u22, 32, [3, 3])

                u11 = slim.conv2d(tf.compat.v1.depth_to_space(input=u23, block_size=2), 16, [3, 3])
                u12 = slim.conv2d(u11, 16, [3, 3])
                u13 = slim.conv2d(u12, 16, [3, 3])

                p = slim.conv2d(u13, 3 if i == 0 else 1, [1, 1],
                                normalizer_fn=None,
                                activation_fn=tf.nn.sigmoid if i == 0 else None)
                if i == 0:
                    rec = p
                else:
                    decoders.append(p)
            seg = tf.concat(decoders, axis=-1)
            return mu, log_sig_sq, rec, seg, z, c_centers, descr


def model_fn(features, labels, mode, hparams):
    del labels

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    sz = FLAGS.batch_size if is_training else FLAGS.batch_size // 3

    im_l = tf.concat([features['img'], features['label']], axis=-1)
    # x = tf.image.random_flip_left_right(im_l)
    x = tf.image.random_crop(im_l, [tf.shape(input=im_l)[0], vh, vw, 3 + N_CLASSES])
    features['img'] = x[:, :, :, :3]
    labels = x[:, :, :, 3:]
    if is_training:
        images = features['img']
    else:
        images = tf.concat([features['img'], features['cl_live'], features['cl_mem']], 0)

    im_warp = tf.image.random_flip_left_right(images)
    im_warp = layers.rand_warp(im_warp, [vh, vw])
    im_w_adj = tf.clip_by_value(im_warp + \
                                tf.random.uniform([tf.shape(input=im_warp)[0], 1, 1, 1], -.8, 0.0),
                                0.0, 1.0)
    tf.compat.v1.where(tf.less(tf.reduce_mean(input_tensor=im_warp, axis=[1, 2, 3]), 0.2), im_warp, im_w_adj)

    mu, log_sig_sq, rec, seg, z, c_centers, descr = vss(images, is_training)
    descr_p = vss(im_warp, is_training, True, True)
    descr_n = utils.hard_neg_mine(descr)

    lp = tf.reduce_sum(input_tensor=descr_p * descr, axis=-1)
    ln = tf.reduce_sum(input_tensor=descr_n * descr, axis=-1)
    m = 0.5
    simloss = tf.reduce_mean(input_tensor=tf.maximum(tf.zeros_like(ln), ln + m - lp))

    # labels = tf.cast(labels, tf.bool)
    # label_ext = tf.concat([tf.expand_dims(labels,-1),
    #            tf.logical_not(tf.expand_dims(labels, -1))], axis=-1)

    if is_training:
        _seg = tf.nn.softmax(seg, axis=-1)
    else:
        _seg = tf.nn.softmax(seg[:FLAGS.batch_size // 3], axis=-1)

    weights = tf.compat.v1.placeholder_with_default(_weights, _weights.shape)
    weights = weights / tf.reduce_min(input_tensor=weights)
    _seg = tf.clip_by_value(_seg, 1e-6, 1.0)
    segloss = tf.reduce_mean(
        input_tensor=-tf.reduce_sum(input_tensor=labels * weights * tf.math.log(_seg), axis=-1))

    recloss = tf.reduce_mean(
        input_tensor=-tf.reduce_sum(input_tensor=images * tf.math.log(tf.clip_by_value(rec, 1e-10, 1.0))
                       + (1.0 - images) * tf.math.log(tf.clip_by_value(1.0 - rec, 1e-10, 1.0)),
                       axis=[1, 2, 3]))

    sh = mu.get_shape().as_list()
    nwh = sh[1] * sh[2] * sh[3]
    m = tf.reshape(mu, [-1, nwh])  # [?, 16 * w*h]
    s = tf.reshape(log_sig_sq, [-1, nwh])
    # stdev is the diagonal of the covariance matrix
    # .5 (tr(sigma2) + mu^T mu - k - log det(sigma2))
    kld = tf.reduce_mean(
        input_tensor=-0.5 * (tf.reduce_sum(input_tensor=1.0 + s - tf.square(m) - tf.exp(s), axis=-1)))

    kld = tf.debugging.check_numerics(kld, '\n\n\n\nkld is inf or nan!\n\n\n')
    recloss = tf.debugging.check_numerics(recloss, '\n\n\n\nrecloss is inf or nan!\n\n\n')
    segloss = tf.debugging.check_numerics(segloss, '\n\n\n\nsegloss is inf or nan!\n\n\n')

    loss = segloss + \
           0.0001 * kld + \
           0.0001 * recloss + \
           simloss

    prob = _seg[0, :, :, :]
    pred = tf.argmax(input=prob, axis=-1)

    mask = tf.argmax(input=labels[0], axis=-1)
    if not is_training:
        dlive = descr[(FLAGS.batch_size // 3):(2 * FLAGS.batch_size // 3)]
        dmem = descr[(2 * FLAGS.batch_size // 3):]

        # Compare each combination of live to mem
        tlive = tf.tile(dlive,
                        [tf.shape(input=dlive)[0], 1])  # [l0, l1, l2..., l0, l1, l2...]

        tmem = tf.reshape(tf.tile(tf.expand_dims(dmem, 1),
                                  [1, tf.shape(input=dlive)[0], 1]),
                          [-1, dlive.get_shape().as_list()[1]])  # [m0, m0, m0..., m1, m1, m1...]

        sim = tf.reduce_sum(input_tensor=tlive * tmem, axis=-1)  # Cosine sim for rgb data + class data
        # Average score across rgb + classes. Map from [-1,1] -> [0,1]
        sim = (1.0 + sim) / 2.0

        sim_sq = tf.reshape(sim,
                            [FLAGS.batch_size // 3, FLAGS.batch_size // 3])

        # Correct location is along diagonal
        labm = tf.reshape(tf.eye(FLAGS.batch_size // 3,
                                 dtype=tf.int64), [-1])

        # ID of nearest neighbor from 
        ids = tf.argmax(input=sim_sq, axis=-1)

        # I guess just contiguously index it?
        row_inds = tf.range(0, FLAGS.batch_size // 3,
                            dtype=tf.int64) * (FLAGS.batch_size // 3 - 1)
        buffer_inds = row_inds + ids
        sim_nn = tf.nn.embedding_lookup(params=sim, ids=buffer_inds)
        # Pull out the labels if it was correct (0 or 1)
        lab = tf.nn.embedding_lookup(params=labm, ids=buffer_inds)

    def touint8(img):
        return tf.cast(img * 255.0, tf.uint8)

    _im = touint8(images[0])
    _rec = touint8(rec[0])

    with tf.compat.v1.variable_scope("stats"):
        tf.compat.v1.summary.scalar("loss", loss)
        tf.compat.v1.summary.scalar("segloss", segloss)
        tf.compat.v1.summary.scalar("kld", kld)
        tf.compat.v1.summary.scalar("recloss", recloss)
        tf.compat.v1.summary.scalar("simloss", simloss)
        tf.compat.v1.summary.histogram("z", z)
        tf.compat.v1.summary.histogram("mu", mu)
        tf.compat.v1.summary.histogram("sig", tf.exp(log_sig_sq))
        tf.compat.v1.summary.histogram("clust_centers", c_centers)

    eval_ops = {
        "Test Error": tf.compat.v1.metrics.mean(loss),
        "Seg Error": tf.compat.v1.metrics.mean(segloss),
        "Rec Error": tf.compat.v1.metrics.mean(recloss),
        "KLD Error": tf.compat.v1.metrics.mean(kld),
        "Sim Error": tf.compat.v1.metrics.mean(simloss),
    }

    if not is_training:
        # Closer to 1 is better
        eval_ops["AUC"] = tf.compat.v1.metrics.auc(lab, sim_nn, curve='PR')
    to_return = {
        "loss": loss,
        "segloss": segloss,
        "recloss": recloss,
        "simloss": simloss,
        "kld": kld,
        "eval_metric_ops": eval_ops,
        'pred': pred,
        'rec': _rec,
        'label': mask,
        'im': _im
    }

    predictions = {
        'pred': seg,
        'rec': rec
    }

    to_return['predictions'] = predictions

    utils.display_trainable_parameters()

    return to_return


def _default_hparams():
    """Returns default or overridden user-specified hyperparameters."""

    hparams = tf.contrib.training.HParams(
        learning_rate=1.0e-3
    )
    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)
    return hparams


def main(argv):
    del argv

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    if FLAGS.mode == 'train':
        hparams = _default_hparams()

        utils.train_and_eval(
            model_dir=FLAGS.model_dir,
            model_fn=model_fn,
            input_fn=create_input_fn,
            hparams=hparams,
            steps=FLAGS.steps,
            batch_size=FLAGS.batch_size,
        )
    elif FLAGS.mode == 'pr':
        import test_net

        test_net.plot(FLAGS.model_dir, FLAGS.data_dir,
                      FLAGS.n_include, FLAGS.title, netvlad_feat=FLAGS.netvlad_feat,
                      include_calc=FLAGS.include_calc)

    elif FLAGS.mode == 'best':
        import test_net

        test_net.find_best_checkpoint(FLAGS.model_dir, FLAGS.data_dir,
                                      FLAGS.n_include)

    elif FLAGS.mode == 'ex':

        utils.show_example(FLAGS.image_fl, FLAGS.model_dir)

    else:
        raise ValueError("Unrecognized mode: %s" % FLAGS.mode)


if __name__ == "__main__":
    sys.excepthook = utils.colored_hook(
        os.path.dirname(os.path.realpath(__file__)))
    tf.compat.v1.app.run()
