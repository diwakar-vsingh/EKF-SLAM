#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.coco_classes import calc_classes
from dataset.gen_tfrecords import vw as __vw
from dataset.gen_tfrecords import vh as __vh

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

import os
import sys
import time
import datetime
import argparse
import utils
import layers
import numpy as np
import tensorflow as tf

tf.keras.backend.clear_session()  # For easy reset of notebook state.

# Define the checkpoint directory to store the checkpoints
checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Create a MirroredStrategy object
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

AUTOTUNE = tf.data.experimental.AUTOTUNE
N_CLASSES = len(calc_classes.keys())
BUFFER_SIZE = 400
path = 'saved_model/'

vw = 256
vh = 192  # Need 128 since we go down by factors of 2

with open('dataset/loss_weights.txt', 'r') as f:
  _weights = np.reshape(np.fromstring(f.read(),
               sep=' ', 
               dtype=np.float32,
               count=N_CLASSES), 
  (1, 1, 1, -1))

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, metavar='', default="train", help="train, pr, ex, or best")
parser.add_argument("-md","--model_dir", type=str, metavar='', default="model", help="Estimator model_dir")
parser.add_argument("-dd","--data_dir", type=str, metavar='', default="dataset/CampusLoopDataset", help="path to data")
parser.add_argument("-t", "--title", type=str, metavar='', default="Precision-Recall Curve", help="Plot title")
parser.add_argument("-n","--n_include", type=int, metavar='', default=5)
parser.add_argument("-s", "--steps", type=int, metavar='', default=200000, help="Training steps")
parser.add_argument("-hp", "--hparams", type=str, metavar='', help="A comma-separated list of `name=value \
                  hyperparameter values. This flag is used to override hyperparameter \
                  settings when manually selecting hyperparameters.")
parser.add_argument("-b", "--batch_size", type=int,  metavar='', default=12, help="Size of mini-batch")
parser.add_argument("-nf", "--netvlad_feat", type=str, metavar='', default=None, help="Binary base file for NetVLAD features. If you did this for dataset XX, \
        the program will look for XX_db.bin and XX_q.bin" )
parser.add_argument("-i", "--input_dir", type=str, metavar='', default="mnt/coco/calc_tfrecords/", help="tfrecords directory")
parser.add_argument("-c", "--include_calc", default=True, action='store_false', help="Include original calc in pr curve \
                     Place in 'calc_model' directory if this is set")
parser.add_argument("-if", "--image_fl", type=str, metavar='')
args = parser.parse_args()


def input_function(split, batch_size):

   """
   input_function for tf.estimator.Estimator
   """

   def decode_features(features):

      # tf.cast:      Casts a tensor to a new type.
      # tf.io.decode_raw: Convert raw byte strings into tensors.
      
      # convert the tf.Example compatible data type to a uint8 tensor
      features = tf.io.decode_raw(features, tf.uint8)
      
      # Cast image data to floats in the [0,1] range.
      features = tf.cast(features, tf.float32)/255.0
      
      # resize the image to the desired size.
      return tf.reshape(features, [__vh, __vw, 3])

      
   def parser(serialized_example):
     '''
     The following function reads a serialized_example and parse it using the feature description.
     '''

     if split == 'train':
      feature_description = {
               'img': tf.io.FixedLenFeature([], tf.string), 
               'label': tf.io.FixedLenFeature([], tf.string)
      } 
     else:
      feature_description = {
               'cl_live': tf.io.FixedLenFeature([], tf.string),
               'cl_mem': tf.io.FixedLenFeature([], tf.string)
      }

     # Parse the input tf.Example proto using the dictionary above.
     fs = tf.io.parse_single_example(serialized=serialized_example, features=feature_description)

     if split == 'train':
      fs['img'] = decode_features(fs['img'])
      fs['label'] = tf.reshape(tf.io.decode_raw(fs['label'], tf.uint8), [__vh, __vw])
      fs['label'] = tf.cast(tf.one_hot(fs['label'], N_CLASSES), tf.float32)
     else:
      fs['cl_live'] = tf.reshape(tf.image.resize(decode_features(fs['cl_live']), (vh, vw)), [vh, vw, 3])
      fs['cl_mem'] = tf.reshape(tf.image.resize(decode_features(fs['cl_live']), (vh, vw)), [vh, vw, 3])
     
     return fs


   def train_preprocess(features):
      """
      Image Pre-processing for training

      Apply the following operations:
         - Concatenate images with their mask along channels
         - Randomly crop to a size of [vh, vw, :]
      """

      # Concatenate images with their mask along channels
      im_l = tf.concat([features['img'], features['label']], axis=-1)

      # Randomly crop image to a size of [vh, vw, 3 + N_CLASSES]
      img = tf.image.random_crop(im_l, size=[vh, vw, 3 + N_CLASSES])
   
      # Extract features and labels back
      features['img'] = img[:, :, :3]
      labels = img[:, :, 3:]

      if split == 'train':
         images = features['img']
      else:
         images = tf.concat([features['img'], features['cl_live'], features['cl_mem']], 0)

      # Randomly flip an image horizontally (left to right).
      im_warp = tf.image.random_flip_left_right(images)
 
      # Randomly warp training images using homographies to emulate camera motion to certain degree
      im_warp = rand_warp(im_warp, [vh, vw])

      # Clip tensor values to a specified min and max.
      im_w_adj = tf.clip_by_value(im_warp + tf.random.uniform(shape=[1, 1, 1], minval=-0.8, maxval=0.0), clip_value_min=0.0, clip_value_max=1.0)
      im_warp = tf.compat.v1.where(tf.math.less(tf.reduce_mean(input_tensor=im_warp, axis=[0, 1, 2]), 0.2), im_warp, im_w_adj)
   
      return ({'Anchor_Image': images, 'Positive_Image': im_warp}, {'Semantics': labels})


   def prepare_for_training(ds, split, batch_size):

      # Shuffle the dataset and Repeat forever
      ds = ds.shuffle(buffer_size=BUFFER_SIZE, seed=np.int64(time.time())).repeat()

      # Prepare batches
      # Use the extra computing power effectively by increasing the batch size
      BATCH_SIZE_PER_REPLICA = batch_size
      global_batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
      ds = ds.batch(global_batch_size if split == 'train' else global_batch_size // 3)

      # `prefetch` lets the dataset fetch batches in the background while the model
      # is training.
      ds = ds.prefetch(buffer_size=AUTOTUNE)

      return ds


   # To load the files as a tf.data.Dataset first create a dataset of the file paths
   # by listing all TFRecord files
   if split == 'train':
      tfrecord = 'train_data*.tfrecord'
      files = tf.data.Dataset.list_files(args.input_dir + tfrecord, shuffle=True, seed=np.int64(time.time()))
      # Disregard data order in favor of reading speed
      ignore_order = tf.data.Options()
      ignore_order.experimental_deterministic = False
      files = files.with_options(ignore_order)

   else:
      tfrecord = 'validation_data.tfrecord'
      files = [args.input_dir + tfrecord]


   # PARALLELIZING DATA EXTRACTION
   # Read the tfrecord as tf.data.Dataset format in interleaved order
   raw_dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)

   # Parse and decode each searilzed tf.train.Example back into Tensor values
   dataset = raw_dataset.map(parser, num_parallel_calls=AUTOTUNE).cache()

   # Data augmentation for the images
   dataset = dataset.map(train_preprocess, num_parallel_calls=AUTOTUNE)

   # Transform it into a new dataset by chaining method calls on the tf.data.Dataset object
   dataset =  prepare_for_training(dataset, split, batch_size)

   return dataset



def VSS(inputShape, alpha=1.0):
   
   ############### Variational Semantic Segmentator ##################
   
   def convolutional_block(name, filters, kernel_size, In):
      x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               padding="same", 
               use_bias=True,
               name=name,
               kernel_initializer=VarianceScaling(scale=1.0, mode='fan_avg',
               distribution=("uniform" if False else "truncated_normal")))(In)
      x = BatchNormalization(axis=-1, 
                        momentum=0.9997, 
                        epsilon=1e-5, 
                        scale=True, 
                        fused=True, 
                        name=name + "_BN")(x)
      x = ELU(alpha=alpha, name=name + "_act")(x)

      return x

   def residual_block(name, filters, kernel_size, In1, In2):
      x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               padding="same", 
               use_bias=True,
               name=name,
               kernel_initializer=VarianceScaling(scale=1.0, mode='fan_avg',
               distribution=("uniform" if False else "truncated_normal")))(In1)
      x = BatchNormalization(axis=-1, 
                        momentum=0.9997, 
                        epsilon=1e-5, 
                        scale=True, 
                        fused=True, 
                        name=name + "_BN")(x)

      x = add([In2, x], name=name + "_sum")
      
      x = ELU(alpha=alpha, name=name+ "_act")(x)

      return x

   def sampling(args):
      """
      Reparameterization trick by sampling from an isotropic unit Gaussian.
      
      # Arguments
        args (tensor): mean and log of variance of Q(z|X)
      
      # Returns
        z (tensor): sampled latent vector
      
      """

      z_mean, z_log_var = args

      # epsilon is a sample from a N(0, 1) distribution
      epsilon = K.random_normal(shape=tf.shape(input=z_mean), dtype=tf.float32)

      return z_mean + K.exp(0.5 * z_log_var) * epsilon


   ########################### Encoder ####################################
   input_layer = Input(shape=inputShape, name='input')
   encoder_input_anchor = Input(shape=inputShape, name='Anchor_Image')
   encoder_input_positive = Input(shape=inputShape, name='Positive_Image')
   
   x = convolutional_block("r1", 32, (3, 3), input_layer)
      
   # Residual Block1 
   x_shortcut = tf.identity(x)
   x = convolutional_block("r2", 16, (1, 1), x)
   x = residual_block("r3", 32, (3, 3), x, x_shortcut)

   # Residual Block2 
   x_shortcut = tf.identity(x)
   x = convolutional_block("r4", 16, (1, 1), x)
   x = residual_block("r5", 32, (3, 3), x, x_shortcut)
   x = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name="p1")(x)
   
   filters = [64, 128, 256]

   # loop over the number of filters
   x = tf.identity(x)
      
   j = 1
   for f in filters:
      j += 1
      x = convolutional_block("d" + str(j) + "1", f, (3, 3), x)
      x = convolutional_block("d" + str(j) + "2", f, (3, 3), x)
      x = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name="p" + str(j))(x)
         
   x = convolutional_block("d51", 512, (3, 3), x)
   x = convolutional_block("d52", 512, (3, 3), x)

   model = Model(input_layer, x)
   anchor = model(encoder_input_anchor)
   pos = model(encoder_input_positive)

   ##################################### Latent vars #######################################

   # Dont slice since we dont want to compute twice as many feature maps for nothing
   z_mean1 = Conv2D(filters=4 * (1 + N_CLASSES), 
               kernel_size=(3, 3), 
               padding="same",
               use_bias=True,
               name="z_mean1",
               kernel_initializer=VarianceScaling(scale=1.0, mode='fan_avg',
               distribution=("uniform" if False else "truncated_normal")))(anchor)


   z_mean2 = Conv2D(filters=4 * (1 + N_CLASSES), 
               kernel_size=(3, 3), 
               padding="same",
               use_bias=True,
               name="z_mean2",
               kernel_initializer=VarianceScaling(scale=1.0, mode='fan_avg',
               distribution=("uniform" if False else "truncated_normal")))(pos)


   z_log_var = Conv2D(filters=4 * (1 + N_CLASSES), 
                     kernel_size=(3, 3), 
                     padding="same", 
                     use_bias=True,
                     name="z_log_var",
                     kernel_initializer=VarianceScaling(scale=1.0, mode='fan_avg',
                     distribution=("uniform" if False else "truncated_normal")))(anchor)


   # Random normal variable for decoder
   z = Lambda(sampling, name='z')([z_mean1, z_log_var])
      
   ####################################### Decoder ##############################################
   decoders = []

   def sem_network(i, sem_input):
      x = convolutional_block("d11" + str(i), 128, (3, 3), sem_input)
      x = tf.compat.v1.depth_to_space(input=x, block_size=2, name="u11" + str(i))
      x = convolutional_block("u12" + str(i), 128, (3, 3), x)
      x = convolutional_block("u13" + str(i), 128, (3, 3), x)

      filters = [64, 32, 16]
      j = 1
         
      for f in filters:
         j = j + 1
         x = tf.compat.v1.depth_to_space(input=x, block_size=2, name=str(j) + str(i) + "dts")
         x = convolutional_block("u" + str(j) + "1" + str(i), f, (3, 3), x)
         x = convolutional_block("u" + str(j) + "2" + str(i), f, (3, 3), x)
         x = convolutional_block("u" + str(j) + "3" + str(i), f, (3, 3), x)

      p = Conv2D(filters=3 if i == 0 else 1, 
                     kernel_size=(1, 1), 
                     padding="same", 
                     use_bias=True,
                     name="rec" if i == 0 else "p" + str(j) + str(i),
                     activation="sigmoid" if i==0 else None,
                     kernel_initializer=VarianceScaling(scale=1.0, mode='fan_avg',
                     distribution=("uniform" if False else "truncated_normal")))(x)

      return p

   j = 0
   for i in range(1 + N_CLASSES):
      val = z[:, :, :, j:j+4]
      j = j + 4

      if i == 0:
         rec = sem_network(i, val)
      else:
         decoders.append(sem_network(i, val))
   
   seg = Concatenate(axis=-1, name="Semantics")(decoders)

   # Instantiate an end-to-end model 
   inputs = [encoder_input_anchor, encoder_input_positive]
   outputs = [seg, rec, z_mean2]
   model = Model(inputs, outputs, name="Autoencoder")

   return z_mean1, z_log_var, z, inputs, outputs, model



# Create and compile the Keras model in the context of strategy.scope.
with strategy.scope():

   def intra_normalization(z_mean, channel_center):
      sh = z_mean.get_shape().as_list()

      # Residuals are intra-normalized using L2 norm across channels to reduce bursts in descriptors
      residual = z_mean - c_centers
      descriptor = K.l2_normalize(residual, axis=-1)
      descriptor = K.reshape(descr, [-1, sh[1] * sh[2] * sh[3]])

      # Normalize the entire descriptor to allow for cosine similarilty calculation by inner product
      descriptor = K.l2_normalize(descriptor, axis=-1)

      return descriptor


   #################################### Variational Semantics Segmentator ####################################

   input_shape = (vh, vw, 3)
   z_mean_anchor, z_log_var, z, inputs, outputs, vss = VSS(inputShape=input_shape)
   images, images_pos = inputs
   seg, rec, z_mean_pos = outputs
   c_centers = tf.Variable(tf.random.normal([1, sh[1], sh[2], sh[3]]), name='offset_anchor', trainable=True)

   ##################################### Kullback-Leibler Divergence ########################################
   sh = z_mean_anchor.get_shape().as_list()
   nwh = sh[1] * sh[2] * sh[3]
   m = tf.reshape(z_mean, [-1, nwh])  # [?, 16 * w*h]
   s = tf.reshape(z_log_var, [-1, nwh])
   kld = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + s - tf.square(m) - tf.exp(s), axis=-1))

   
   ######################################### Siamese network ###################################################

   ## Triplets
   descr = intra_normalization(z_mean_anchor, c_centers)
   descr_p = intra_normalization(z_mean_pos, c_centers)
   descr_n = utils.hard_neg_mine(descr)    

   lp = tf.reduce_sum(input_tensor=descr_p * descr, axis=-1)
   ln = tf.reduce_sum(input_tensor=descr_n * descr, axis=-1)
   m = 0.5

   # Triplet embedding objective function 
   simloss = tf.reduce_mean(input_tensor=tf.maximum(tf.zeros_like(ln), m + ln - lp))


   #################################### Semantics Segmentation Network ###############################################
   def semantics_loss(y_true, y_pred):

      if args.mode == "train":
         y_pred = Softmax(axis=-1)(y_pred)
      else:
         y_pred = Softmax(axis=-1)(y_pred[:args.batch_size // 3])

      weights = _weights / tf.reduce_min(input_tensor=_weights)
      y_pred = K.clip(y_pred, min_value=1e-6, max_value=1.0)
      return tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y_true * weights * tf.math.log(y_pred), axis=-1))


   #################################### Reconstructed full RGB image Loss #######################################
   recloss = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=images * K.log(K.clip(rec, 1e-10, 1.0)) + 
                                          (1.0 - images) * K.log(K.clip(1.0 - rec, 1e-10, 1.0)), axis=[1, 2, 3]))

   
   # Checks a tensor for NaN and Inf values.
   tf.debugging.check_numerics(kld, '\n\n\n\nkld is inf or NaN!\n\n\n')
   tf.debugging.check_numerics(simloss, '\n\n\n\nsimloss is inf or NaN!\n\n\n')
   tf.debugging.check_numerics(recloss, '\n\n\n\nrecloss is inf or NaN!\n\n\n')


   ##################################### The overall objective function (Loss) ####################################
   loss = 0.0001*kld + 0.0001*recloss + simloss

   vss.add_loss(loss)
   vss.add_metric(tf.keras.metrics.Mean(loss, name="Total_Loss"), aggregation='mean')
   vss.add_metric(tf.keras.metrics.Mean(recloss, name="RGB_Reconstruction_Loss"), aggregation='mean')
   vss.add_metric(tf.keras.metrics.Mean(kld, name="KL_Divergence"), aggregation='mean')
   vss.add_metric(tf.keras.metrics.Mean(simloss, name="Triplet_Embedding_Loss"), aggregation='mean')

   vss.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, name="Adam"),
            loss={"Semantics": semantics_loss},
            metrics={"Semantics": [tf.keras.metrics.Mean(name="Semantics_Segmentation_Loss")]})

   log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

   # Callbacks
   callbacks = [
         tf.keras.callbacks.TensorBoard(log_dir='log_dir', histogram_freq=1),
         tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
   ]

   # Load dataset
   dataset = input_function("train", args.batch_size)
   val_dataset = input_function("val", args.batch_size)
   vss.fit(dataset, 
         steps_per_epoch=200,
         callbacks=callbacks,
         validation_data=val_dataset)

   model.save(path, save_format='tf')

   _im = tf.cast(images[0]*255.0, tf.uint8)
   _rec = tf.cast(rec[0]*255.0, tf.uint8) 

   prob = _seg[0, :, :, :]
   pred = tf.argmax(input=prob, axis=-1)
   mask = tf.argmax(input=labels[0], axis=-1)
      
   if not is_training:
      dlive = descr[(args.batch_size // 3):(2 * args.batch_size // 3)]
      dmem = descr[(2 * args.batch_size // 3):]

      # Compare each combination of live to mem
      tlive = tf.tile(dlive, [tf.shape(input=dlive)[0], 1])  # [l0, l1, l2..., l0, l1, l2...]

      tmem = tf.reshape(tf.tile(tf.expand_dims(dmem, 1), [1, tf.shape(input=dlive)[0], 1]),
                  [-1, dlive.get_shape().as_list()[1]])  # [m0, m0, m0..., m1, m1, m1...]

      sim = tf.reduce_sum(input_tensor=tlive * tmem, axis=-1)  # Cosine sim for rgb data + class data

      # Average score across rgb + classes. Map from [-1,1] -> [0,1]
      sim = (1.0 + sim) / 2.0

      sim_sq = tf.reshape(sim, [args.batch_size // 3, args.batch_size // 3])

      # Correct location is along diagonal
      labm = tf.reshape(tf.eye(args.batch_size // 3, dtype=tf.int64), [-1])

      # ID of nearest neighbor from 
      ids = tf.argmax(input=sim_sq, axis=-1)

      # I guess just contiguously index it?
      row_inds = tf.range(0, args.batch_size // 3, dtype=tf.int64) * (args.batch_size // 3 - 1)
      buffer_inds = row_inds + ids
      sim_nn = tf.nn.embedding_lookup(params=sim, ids=buffer_inds)

      # Pull out the labels if it was correct (0 or 1)
      lab = tf.nn.embedding_lookup(params=labm, ids=buffer_inds)

      # Closer to 1 is better
      vss.add_metric(tf.keras.metrics.AUC(curve='PR'), aggregation='mean')

      eval_ops["AUC"] = tf.compat.v1.metrics.auc(labels=lab, predictions=sim_nn, curve='PR')

   utils.display_trainable_parameters()



def main():

   # Sets the threshold for what messages will be logged.
   tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

   with strategy.scope():
      def intra_normalization(z_mean, channel_center):
         sh = z_mean.get_shape().as_list()

         # Residuals are intra-normalized using L2 norm across channels to reduce bursts in descriptors
         residual = z_mean - c_centers
         descriptor = K.l2_normalize(residual, axis=-1)
         descriptor = K.reshape(descr, [-1, sh[1] * sh[2] * sh[3]])

         # Normalize the entire descriptor to allow for cosine similarilty calculation by inner product
         descriptor = K.l2_normalize(descriptor, axis=-1)

         return descriptor


      #################################### Variational Semantics Segmentator ####################################

      input_shape = (vh, vw, 3)
      z_mean_anchor, z_log_var, z, inputs, outputs, vss = VSS(inputShape=input_shape)
      images, images_pos = inputs
      seg, rec, z_mean_pos = outputs
      c_centers = tf.Variable(tf.random.normal([1, sh[1], sh[2], sh[3]]), name='offset_anchor', trainable=True)

      ##################################### Kullback-Leibler Divergence ########################################
      sh = z_mean_anchor.get_shape().as_list()
      nwh = sh[1] * sh[2] * sh[3]
      m = tf.reshape(z_mean, [-1, nwh])  # [?, 16 * w*h]
      s = tf.reshape(z_log_var, [-1, nwh])
      kld = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + s - tf.square(m) - tf.exp(s), axis=-1))

   
      ######################################### Siamese network ###################################################

      ## Triplets
      descr = intra_normalization(z_mean_anchor, c_centers)
      descr_p = intra_normalization(z_mean_pos, c_centers)
      descr_n = utils.hard_neg_mine(descr)    

      lp = tf.reduce_sum(input_tensor=descr_p * descr, axis=-1)
      ln = tf.reduce_sum(input_tensor=descr_n * descr, axis=-1)
      m = 0.5

      # Triplet embedding objective function 
      simloss = tf.reduce_mean(input_tensor=tf.maximum(tf.zeros_like(ln), m + ln - lp))


      #################################### Semantics Segmentation Network ###############################################
      def semantics_loss(y_true, y_pred):

         if args.mode == "train":
            y_pred = Softmax(axis=-1)(y_pred)
         else:
            y_pred = Softmax(axis=-1)(y_pred[:args.batch_size // 3])

         weights = _weights / tf.reduce_min(input_tensor=_weights)
         y_pred = K.clip(y_pred, min_value=1e-6, max_value=1.0)
         return tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y_true * weights * tf.math.log(y_pred), axis=-1))


      #################################### Reconstructed full RGB image Loss #######################################
      recloss = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=images * K.log(K.clip(rec, 1e-10, 1.0)) + 
                                          (1.0 - images) * K.log(K.clip(1.0 - rec, 1e-10, 1.0)), axis=[1, 2, 3]))

   
      # Checks a tensor for NaN and Inf values.
      tf.debugging.check_numerics(kld, '\n\n\n\nkld is inf or NaN!\n\n\n')
      tf.debugging.check_numerics(simloss, '\n\n\n\nsimloss is inf or NaN!\n\n\n')
      tf.debugging.check_numerics(recloss, '\n\n\n\nrecloss is inf or NaN!\n\n\n')


      ##################################### The overall objective function (Loss) ####################################
      loss = 0.0001*kld + 0.0001*recloss + simloss

      vss.add_loss(loss)
      vss.add_metric(tf.keras.metrics.Mean(loss, name="Total_Loss"), aggregation='mean')
      vss.add_metric(tf.keras.metrics.Mean(recloss, name="RGB_Reconstruction_Loss"), aggregation='mean')
      vss.add_metric(tf.keras.metrics.Mean(kld, name="KL_Divergence"), aggregation='mean')
      vss.add_metric(tf.keras.metrics.Mean(simloss, name="Triplet_Embedding_Loss"), aggregation='mean')

      vss.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, name="Adam"),
            loss={"Semantics": semantics_loss},
            metrics={"Semantics": [tf.keras.metrics.Mean(name="Semantics_Segmentation_Loss")]})

      log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

      # Callbacks
      callbacks = [
         tf.keras.callbacks.TensorBoard(log_dir='log_dir', histogram_freq=1),
         tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
      ]

      # Load dataset
      dataset = input_function("train", args.batch_size)
      val_dataset = input_function("val", args.batch_size)
      vss.fit(dataset, 
         steps_per_epoch=200,
         callbacks=callbacks,
         validation_data=val_dataset)

      model.save(path, save_format='tf')

      _im = tf.cast(images[0]*255.0, tf.uint8)
      _rec = tf.cast(rec[0]*255.0, tf.uint8) 

      prob = _seg[0, :, :, :]
      pred = tf.argmax(input=prob, axis=-1)
      mask = tf.argmax(input=labels[0], axis=-1)
      
      if not is_training:
         dlive = descr[(args.batch_size // 3):(2 * args.batch_size // 3)]
         dmem = descr[(2 * args.batch_size // 3):]

         # Compare each combination of live to mem
         tlive = tf.tile(dlive, [tf.shape(input=dlive)[0], 1])  # [l0, l1, l2..., l0, l1, l2...]

         tmem = tf.reshape(tf.tile(tf.expand_dims(dmem, 1), [1, tf.shape(input=dlive)[0], 1]),
                  [-1, dlive.get_shape().as_list()[1]])  # [m0, m0, m0..., m1, m1, m1...]

         sim = tf.reduce_sum(input_tensor=tlive * tmem, axis=-1)  # Cosine sim for rgb data + class data

         # Average score across rgb + classes. Map from [-1,1] -> [0,1]
         sim = (1.0 + sim) / 2.0

         sim_sq = tf.reshape(sim, [args.batch_size // 3, args.batch_size // 3])

         # Correct location is along diagonal
         labm = tf.reshape(tf.eye(args.batch_size // 3, dtype=tf.int64), [-1])

         # ID of nearest neighbor from 
         ids = tf.argmax(input=sim_sq, axis=-1)

         # I guess just contiguously index it?
         row_inds = tf.range(0, args.batch_size // 3, dtype=tf.int64) * (args.batch_size // 3 - 1)
         buffer_inds = row_inds + ids
         sim_nn = tf.nn.embedding_lookup(params=sim, ids=buffer_inds)

         # Pull out the labels if it was correct (0 or 1)
         lab = tf.nn.embedding_lookup(params=labm, ids=buffer_inds)

         # Closer to 1 is better
         vss.add_metric(tf.keras.metrics.AUC(curve='PR'), aggregation='mean')

         eval_ops["AUC"] = tf.compat.v1.metrics.auc(labels=lab, predictions=sim_nn, curve='PR')

      utils.display_trainable_parameters()



if __name__ == "__main__":
   sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
   main()
