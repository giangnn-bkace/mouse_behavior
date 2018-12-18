# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:40:31 2018

@author: Guest NGN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

class Unit_2D1D(snt.AbstractModule):
    
    
    def __init__(self,
                 in_filters,
                 out_filters,
                 kernels,
                 name):
        
        super(Unit_2D1D, self).__init__(name=name)
        self._in_filters = in_filters
        self._out_filters = out_filters
        self._kernels = kernels
        if self._in_filters != self._out_filters:
            self._strides = [2,2,2]
        else:
            self._strides = [1,1,1]
        
    def _build(self, inputs, is_training):
        
        i = 3 * self._in_filters * self._out_filters * self._kernels[1] * self._kernels[2]
        i /= self._in_filters * self._kernels[1] * self._kernels[2] + 3 * self._out_filters
        middle_filters = int(i)
        net = snt.Conv3D(output_channels=middle_filters,
                     kernel_shape=[1,self._kernels[1],self._kernels[2]],
                     stride=[1,self._strides[1],self._strides[2]],
                     padding=snt.SAME,
                     use_bias=False,
                     name='conv_middle')(inputs)
        
        net = tf.layers.batch_normalization(net, training=is_training, name='spatbn_middle')
        net = tf.nn.relu(net)
        
        net = snt.Conv3D(output_channels=self._out_filters,
                     kernel_shape=[self._kernels[0],1,1],
                     stride=[self._strides[0],1,1],
                     padding=snt.SAME,
                     use_bias=False,
                     name='conv')(net)
        return net
        
        

class R21D_Block(snt.AbstractModule):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 name):
        
        super(R21D_Block, self).__init__(name=name)
        self._in_channels = in_channels
        self._out_channels = out_channels
        
    def _build(self, inputs, is_training):
        
        shortcut = inputs
        
        net = Unit_2D1D(self._in_channels, 
                        self._out_channels, 
                        kernels=[3,3,3],
                        name='1')(inputs, is_training)
        
        net = tf.layers.batch_normalization(net, training=is_training, name='spatbn_1')
        net = tf.nn.relu(net)
        
        net = Unit_2D1D(self._out_channels, 
                        self._out_channels, 
                        kernels=[3,3,3],
                        name='2')(net, is_training)
        
        net = tf.layers.batch_normalization(net, training=is_training, name='spatbn_2')
        
        if self._in_channels != self._out_channels:
            shortcut = snt.Conv3D(output_channels=self._out_channels,
                                  kernel_shape=[1,1,1],
                                  stride=[2,2,2],
                                  padding=snt.SAME,
                                  use_bias=False,
                                  name='shortcut_projection')(shortcut)
            shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name='shortcut_projection_spatbn')
        
        return tf.nn.relu(net+shortcut)

class R2Plus1D(snt.AbstractModule):
    
    VALID_ENDPOINTS = (
            'Logits',
            'Predictions',
    )
    
    
    
    def __init__(self,
                 final_spatial_kernel=7,
                 final_temporal_kernel=1):
        
        super(R2Plus1D, self).__init__(name='R2Plus1D')
        self._final_spatial_kernel = final_spatial_kernel
        self._final_temporal_kernel = final_temporal_kernel
        
    
    
    def _build(self,
               inputs,
               is_training):
        
        net = inputs
        
        ### Decomposition
        net = snt.Conv3D(output_channels=45,
                     kernel_shape=[1,7,7],
                     stride=[1,2,2],
                     padding=snt.SAME,
                     use_bias=False,
                     name='conv1_middle')(net)
        
        net = tf.layers.batch_normalization(net, training=is_training, name='conv1_middle_spatbn_relu')
        
        net = tf.nn.relu(net)
        
        net = snt.Conv3D(output_channels=64,
                     kernel_shape=[3,1,1],
                     stride=[1,1,1],
                     padding=snt.SAME,
                     use_bias=False,
                     name='conv1')(net)
        
        net = tf.layers.batch_normalization(net, training=is_training, name='conv1_spatbn_relu')
        
        net = tf.nn.relu(net)
        
        # conv_2x
        net = R21D_Block(64, 64, name='comp_0')(net, is_training)
        net = R21D_Block(64, 64, name='comp_1')(net, is_training)
        net = R21D_Block(64, 64, name='comp_2')(net, is_training)
        
        # conv_3x
        net = R21D_Block(64, 128, name='comp_3')(net, is_training)
        net = R21D_Block(128, 128, name='comp_4')(net, is_training)
        net = R21D_Block(128, 128, name='comp_5')(net, is_training)
        net = R21D_Block(128, 128, name='comp_6')(net, is_training)
        
        # conv_4x
        net = R21D_Block(128, 256, name='comp_7')(net, is_training)
        net = R21D_Block(256, 256, name='comp_8')(net, is_training)
        net = R21D_Block(256, 256, name='comp_9')(net, is_training)
        net = R21D_Block(256, 256, name='comp_10')(net, is_training)
        net = R21D_Block(256, 256, name='comp_11')(net, is_training)
        net = R21D_Block(256, 256, name='comp_12')(net, is_training)
        
        #conv_5x
        net = R21D_Block(256, 512, name='comp_13')(net, is_training)
        net = R21D_Block(512, 512, name='comp_14')(net, is_training)
        net = R21D_Block(512, 512, name='comp_15')(net, is_training)
        
        #Final layers
        #print(net.shape)
        net = tf.nn.pool(net,
                         window_shape=[
                                 self._final_temporal_kernel,
                                 self._final_spatial_kernel,
                                 self._final_spatial_kernel
                         ],
                         pooling_type="AVG",
                         strides=[1,1,1],
                         padding='VALID')
        logits = tf.squeeze(net, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)
        
        return averaged_logits
        
        