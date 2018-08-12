# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

import scipy.io as sio
import numpy as np

FLAGS = flags.FLAGS

class NetVLAD_NonLocal_types():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        a = tf.multiply(a_sum,cluster_weights2)

        activation = tf.transpose(activation,perm=[0,2,1])

        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)

        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.reshape(vlad, [-1, self.feature_size])

        vlad_softmax = self.embedgaussian_relation(vlad, 1/float(64))


        nonlocal_g = tf.get_variable("nonlocal_g",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        nonlocal_out = tf.get_variable("nonlocal_out",
              [self.cluster_size, self.feature_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)))

        vlad_g = tf.matmul(vlad, nonlocal_g)
        vlad_g = tf.reshape(vlad_g, [-1, self.cluster_size, self.cluster_size])
        vlad_g = tf.matmul(vlad_softmax, vlad_g)
        vlad_g = tf.reshape(vlad_g, [-1, self.cluster_size])

        vlad_g = tf.matmul(vlad_g, nonlocal_out)
        vlad_g = tf.reshape(vlad_g, [-1, self.cluster_size, self.feature_size])
        vlad = tf.reshape(vlad, [-1, self.cluster_size, self.feature_size])
        vlad = vlad + vlad_g

        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1) # [b,f,c]

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

    def embedgaussian_relation(self, input_, temp=1/float(32)):
      nonlocal_theta = tf.get_variable("nonlocal_theta",
            [self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
      nonlocal_phi = tf.get_variable("nonlocal_phi",
            [self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

      vlad_theta = tf.matmul(input_, nonlocal_theta)
      vlad_phi = tf.matmul(input_, nonlocal_phi)
      vlad_theta = tf.reshape(vlad_theta, [-1, self.cluster_size, self.cluster_size])
      vlad_phi = tf.reshape(vlad_phi, [-1, self.cluster_size, self.cluster_size])
      vlad_softmax = tf.nn.softmax(temp * tf.matmul(vlad_theta, tf.transpose(vlad_phi,perm=[0,2,1])))
      return vlad_softmax

class NetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = 300
    add_batch_norm = True
    random_frames = True
    cluster_size = 64
    hidden1_size = 1024
    relu = False
    dimred = -1
    gating = True
    remove_diag = False

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)


    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])


    video_NetVLAD = NetVLAD_NonLocal_types(1024,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetVLAD = NetVLAD_NonLocal_types(128,max_frames,cluster_size/2, add_batch_norm, is_training)


    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024])

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

    vlad = tf.concat([vlad_video, vlad_audio],1)

    vlad_dim = vlad.get_shape().as_list()[1]
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    if relu:
      activation = tf.nn.relu6(activation)


    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))

        gates = tf.matmul(activation, gating_weights)

        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)


        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               'willow_MoeModel')


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class GruModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of GRUs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    gru_size = 1200
    number_of_layers = 2
    backward = False
    random_frames = False
    iterations = 30

    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)

    if backward:
        model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1)

    stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               'MoeModel')
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)



class SoftDBoF():
    def __init__(self, feature_size,max_frames,cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)

        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation,1)
        activation_sum = tf.nn.l2_normalize(activation_sum,1)

        if max_pool:
            activation_max = tf.reduce_max(activation,1)
            activation_max = tf.nn.l2_normalize(activation_max,1)
            activation = tf.concat([activation_sum,activation_max],1)
        else:
            activation = activation_sum

        return activation


class LightVLAD_nonlocal():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        activation = tf.transpose(activation,perm=[0,2,1])

        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)

        vlad = tf.reshape(vlad, [-1,self.feature_size])
        vlad = nonLocal_block(vlad, feature_size=self.feature_size, hidden_size=self.feature_size//2, cluster_size=self.cluster_size)

        vlad = tf.reshape(vlad, [-1,self.cluster_size,self.feature_size])
        vlad = tf.transpose(vlad,perm=[0,2,1])

        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class LightNetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = 300
    add_batch_norm = True
    random_frames = True
    cluster_size = 64
    hidden1_size = 1024
    relu = False
    dimred = -1
    gating = True
    remove_diag = False

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)


    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])


    video_NetVLAD = LightVLAD_nonlocal(1024,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetVLAD = LightVLAD_nonlocal(128,max_frames,cluster_size/2, add_batch_norm, is_training)


    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024])

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

    vlad = tf.concat([vlad_video, vlad_audio],1)

    vlad_dim = vlad.get_shape().as_list()[1]
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    if relu:
      activation = tf.nn.relu6(activation)


    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))

        gates = tf.matmul(activation, gating_weights)

        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)


        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               'willow_MoeModel_moe4')


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

def nonLocal_block(vlad, feature_size, hidden_size, cluster_size):
    nonlocal_theta = tf.get_variable("nonlocal_theta",
          [feature_size, hidden_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    nonlocal_phi = tf.get_variable("nonlocal_phi",
          [feature_size, hidden_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    nonlocal_g = tf.get_variable("nonlocal_g",
          [feature_size, hidden_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    nonlocal_out = tf.get_variable("nonlocal_out",
          [hidden_size, feature_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden_size)))

    vlad_theta = tf.matmul(vlad, nonlocal_theta)
    vlad_phi = tf.matmul(vlad, nonlocal_phi)
    vlad_g = tf.matmul(vlad, nonlocal_g)

    vlad_theta = tf.reshape(vlad_theta, [-1, cluster_size, hidden_size])
    vlad_phi = tf.reshape(vlad_phi, [-1, cluster_size, hidden_size])
    vlad_g = tf.reshape(vlad_phi, [-1, cluster_size, hidden_size])

    vlad_softmax = tf.nn.softmax(feature_size**-.5 * tf.matmul(vlad_theta, tf.transpose(vlad_phi,perm=[0,2,1])))
    vlad_g = tf.matmul(vlad_softmax, vlad_g)
    vlad_g = tf.reshape(vlad_g, [-1, hidden_size])

    vlad_g = tf.matmul(vlad_g, nonlocal_out)
    vlad = vlad + vlad_g
    return vlad

class SoftDbofModelLF(models.BaseModel):
  """Creates a Soft Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = 300
    add_batch_norm = True
    random_frames = True
    cluster_size = 4000
    hidden1_size = 1024
    fc_dimred = True
    relu = False
    max_pool = False

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_Dbof = SoftDBoF(1024,max_frames,cluster_size, max_pool, add_batch_norm, is_training)
    audio_Dbof = SoftDBoF(128,max_frames,cluster_size/8, max_pool, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBOF"):
        dbof_video = video_Dbof.forward(reshaped_input[:,0:1024])

    with tf.variable_scope("audio_DBOF"):
        dbof_audio = audio_Dbof.forward(reshaped_input[:,1024:])

    dbof = tf.concat([dbof_video, dbof_audio],1)

    dbof_dim = dbof.get_shape().as_list()[1]

    if fc_dimred:
        hidden1_weights = tf.get_variable("hidden1_weights",
          [dbof_dim, hidden1_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(dbof, hidden1_weights)

        if add_batch_norm and relu:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="hidden1_bn")
        else:
          hidden1_biases = tf.get_variable("hidden1_biases",
            [hidden1_size],
            initializer = tf.random_normal_initializer(stddev=0.01))
          tf.summary.histogram("hidden1_biases", hidden1_biases)
          activation += hidden1_biases

        if relu:
          activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)
    else:
        activation = dbof

    aggregated_model = getattr(video_level_models,
                               'willow_MoeModel_moe2_noGP')


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class early_NetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = 300
    add_batch_norm = True
    random_frames = True
    cluster_size = 64
    hidden1_size = 1024
    relu = False
    dimred = -1
    gating = True
    remove_diag = False

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)


    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    video_audio_NetVLAD = NetVLAD_NonLocal(1024+128,max_frames,cluster_size, add_batch_norm, is_training)

    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")
    with tf.variable_scope("video_audio_VLAD"):
        vlad = video_audio_NetVLAD.forward(reshaped_input)

    vlad_dim = vlad.get_shape().as_list()[1]
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    if relu:
      activation = tf.nn.relu6(activation)


    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))

        gates = tf.matmul(activation, gating_weights)

        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)


        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               'willow_MoeModel_moe2')


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

class NetVLAD_NonLocal():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        a = tf.multiply(a_sum,cluster_weights2)

        activation = tf.transpose(activation,perm=[0,2,1])

        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)


        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.reshape(vlad, [-1, self.feature_size])

        nonlocal_theta = tf.get_variable("nonlocal_theta",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        nonlocal_phi = tf.get_variable("nonlocal_phi",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        nonlocal_g = tf.get_variable("nonlocal_g",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        nonlocal_out = tf.get_variable("nonlocal_out",
              [self.cluster_size, self.feature_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)))

        vlad_theta = tf.matmul(vlad, nonlocal_theta)
        vlad_phi = tf.matmul(vlad, nonlocal_phi)
        vlad_g = tf.matmul(vlad, nonlocal_g)

        vlad_theta = tf.reshape(vlad_theta, [-1, self.cluster_size, self.cluster_size])
        vlad_phi = tf.reshape(vlad_phi, [-1, self.cluster_size, self.cluster_size])
        vlad_g = tf.reshape(vlad_phi, [-1, self.cluster_size, self.cluster_size])

        vlad_softmax = tf.nn.softmax(self.feature_size**-.5 * tf.matmul(vlad_theta, tf.transpose(vlad_phi,perm=[0,2,1])))
        vlad_g = tf.matmul(vlad_softmax, vlad_g)
        vlad_g = tf.reshape(vlad_g, [-1, self.cluster_size])

        vlad_g = tf.matmul(vlad_g, nonlocal_out)
        vlad_g = tf.reshape(vlad_g, [-1, self.cluster_size, self.feature_size])
        vlad = tf.reshape(vlad, [-1, self.cluster_size, self.feature_size])
        vlad = vlad + vlad_g

        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1) # [b,f,c]

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad


class SoftDbofModelLF_8k(models.BaseModel):
  """Creates a Soft Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = 300
    add_batch_norm = True
    random_frames = True
    cluster_size = 8000
    hidden1_size = 1024
    fc_dimred = True
    relu = False
    max_pool = False

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_Dbof = SoftDBoF(1024,max_frames,cluster_size, max_pool, add_batch_norm, is_training)
    audio_Dbof = SoftDBoF(128,max_frames,cluster_size/8, max_pool, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBOF"):
        dbof_video = video_Dbof.forward(reshaped_input[:,0:1024])

    with tf.variable_scope("audio_DBOF"):
        dbof_audio = audio_Dbof.forward(reshaped_input[:,1024:])

    dbof = tf.concat([dbof_video, dbof_audio],1)

    dbof_dim = dbof.get_shape().as_list()[1]

    if fc_dimred:
        hidden1_weights = tf.get_variable("hidden1_weights",
          [dbof_dim, hidden1_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(dbof, hidden1_weights)

        if add_batch_norm and relu:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="hidden1_bn")
        else:
          hidden1_biases = tf.get_variable("hidden1_biases",
            [hidden1_size],
            initializer = tf.random_normal_initializer(stddev=0.01))
          tf.summary.histogram("hidden1_biases", hidden1_biases)
          activation += hidden1_biases

        if relu:
          activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)
    else:
        activation = dbof

    aggregated_model = getattr(video_level_models,
                               'willow_MoeModel_moe2_noGP')


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
