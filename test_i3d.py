#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
import argparse
import time
import logging

import numpy as np
import tensorflow as tf
import pandas as pd

import i3d
from lib.action_dataset import Action_Dataset
from lib.action_dataset import get_each_frame_test_info

_GLOBAL_EPOCH = 1
_BATCH_SIZE = 1

_CLIP_SIZE = 16
# How many frames are used for each video in testing phase
_FRAME_SIZE = 224

_PREFETCH_BUFFER_SIZE = 50
_NUM_PARALLEL_CALLS = 2

_LOG_ROOT = 'output_momentum_16_run-02'

_MIX_WEIGHT_OF_RGB = 0.2
_MIX_WEIGHT_OF_FLOW = 0.8

_CHECKPOINT_PATHS = {
    'rgb': './data/checkpoints/rgb_scratch/model.ckpt',
    'flow': './data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': './data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': './data/checkpoints/flow_imagenet/model.ckpt',
}

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51,
    'clipped_data': 8
}

# NOTE: Before running, change the path of data
_DATA_ROOT = {
    'ucf101': {
        'rgb': '/data1/yunfeng/dataset/ucf101/jpegs_256',
        'flow': '/data1/yunfeng/dataset/ucf101/tvl1_flow/{:s}'
    },
    'hmdb51': {
        'rgb': '/data2/yunfeng/dataset/hmdb51/jpegs_256',
        'flow': '/data2/yunfeng/dataset/hmdb51/tvl1_flow/{:s}'
    },
    'clipped_data': {
        #'rgb': os.path.join('data', 'clipped_data', 'rgb'),
        #'flow': os.path.join('data', 'clipped_data', 'tvl1', 'flow-{:s}')]
        'rgb': '',
        'flow': ''
    }
}


_CHECKPOINT_PATHS_RGB = [
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-1', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-2', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-3', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-4', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-5', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-6', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-7', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-8', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-9', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-10', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-11', 'clipped_data_rgb-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-rgb-12', 'clipped_data_rgb-90')
]
_CHECKPOINT_PATHS_FLOW = [
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-1', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-2', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-3', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-4', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-5', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-6', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-7', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-8', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-9', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-10', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-11', 'clipped_data_flow-90'),
    os.path.join(_LOG_ROOT, 'finetune-clipped_data-flow-12', 'clipped_data_flow-90')
]

def _get_data_label_from_info(rgb_info_tensor, flow_info_tensor, name):
    """ Wrapper for `tf.py_func`, get video clip and label from info list."""
    rgb_clip_holder, rgb_label_holder = tf.py_func(
        process_video, [rgb_info_tensor, name, 'rgb'], [tf.float32, tf.int32])
    flow_clip_holder, flow_label_holder = tf.py_func(
        process_video, [flow_info_tensor, name, 'flow'], [tf.float32, tf.int32])
    return rgb_clip_holder, rgb_label_holder, flow_clip_holder, flow_label_holder


def process_video(data_info, name, mode, is_training=False):
    """ Get video clip and label from data info list."""
    data = Action_Dataset(name, mode, [data_info])
    if is_training:
        clip_seq, label_seq = data.next_batch(1, _CLIP_SIZE)
    else:
        clip_seq, label_seq = data.get_element(1, _CLIP_SIZE)
    clip_seq = 2*(clip_seq/255) - 1
    clip_seq = np.array(clip_seq, dtype='float32')
    return clip_seq, label_seq


def main(dataset='ucf101', mode='mixed', split=1):
    assert mode in ['rgb', 'flow', 'mixed'], 'Only RGB data and flow data is supported'
    log_dir = os.path.join(_LOG_ROOT, 'test')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'log-%d.txt' % (split)),
                        filemode='w', format='%(message)s')

    ##  Data Preload  ###
    rgb_test_info = get_each_frame_test_info(
        os.path.join('./data', dataset,'rgb.csv'),
        os.path.join('./data', dataset, 'testlist%02d' % split+'.txt'),
        mode='rgb')
#        os.path.join('/data1/yunfeng/i3d_test/data', dataset, mode+'.txt'),
#        os.path.join('/data1/yunfeng/i3d_test/data', dataset, 'testlist%02d' % split+'.txt'))
    flow_test_info = get_each_frame_test_info(
        os.path.join('./data', dataset,'flow.csv'),
        os.path.join('./data', dataset, 'testlist%02d' % split+'.txt'),
        mode='flow')
#        os.path.join('/data1/yunfeng/i3d_test/data', dataset, mode+'.txt'),
#        os.path.join('/data1/yunfeng/i3d_test/data', dataset, 'testlist%02d' % split+'.txt'))
    rgb_data = Action_Dataset(dataset, mode, rgb_test_info)
    flow_data = Action_Dataset(dataset, mode, flow_test_info)
    
    
    
    num_rgb_sample = len(rgb_test_info)
    num_flow_sample = len(flow_test_info)
    print(num_rgb_sample)
    print(rgb_data.size)
    print(num_flow_sample)
    print(flow_data.size)
    
    # Every element in train_info is shown as below:
    # ['v_ApplyEyeMakeup_g08_c01',
    # '/data4/zhouhao/dataset/ucf101/jpegs_256/v_ApplyEyeMakeup_g08_c01',
    # '121', '0']
    #print(rgb_test_info)
    rgb_info_tensor = tf.constant(rgb_test_info)
    flow_info_tensor = tf.constant(flow_test_info)

    # Dataset building
    # Phase 1 Trainning
    # one element in this dataset is (train_info list)
    rgb_info_dataset = tf.data.Dataset.from_tensor_slices(
        (rgb_info_tensor))
    flow_info_dataset = tf.data.Dataset.from_tensor_slices(
        (flow_info_tensor))
    
    test_dataset = tf.data.Dataset.zip((rgb_info_dataset, flow_info_dataset))
    # one element in this dataset is (single image_postprocess, single label)
    test_dataset = test_dataset.map(lambda x, y: _get_data_label_from_info(
        x, y, dataset), num_parallel_calls=_NUM_PARALLEL_CALLS)
    # one element in this dataset is (batch image_postprocess, batch label)
    test_dataset = test_dataset.repeat().batch(_BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    # iterator = dataset.make_one_shot_iterator()
    # clip_holder, label_holder = iterator.get_next()
    iterator = tf.data.Iterator.from_structure(
        test_dataset.output_types, test_dataset.output_shapes)
    test_init_op = iterator.make_initializer(test_dataset)
    
    rgb_clip_holder, rgb_label_holder, flow_clip_holder, flow_label_holder = iterator.get_next()
    
    rgb_clip_holder = tf.squeeze(rgb_clip_holder,  [1])
    rgb_label_holder = tf.squeeze(rgb_label_holder, [1])
    rgb_clip_holder.set_shape(
        [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['rgb']])
    
    flow_clip_holder = tf.squeeze(flow_clip_holder,  [1])
    flow_label_holder = tf.squeeze(flow_label_holder, [1])
    flow_clip_holder.set_shape(
        [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['flow']])
    
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)

    # inference module
    # Inference Module
    with tf.variable_scope(_SCOPE['rgb']):
        # insert i3d model
        rgb_model = i3d.InceptionI3d(
            400, spatial_squeeze=True, final_endpoint='Logits')
        # the line below outputs the final results with logits
        # __call__ uses _template, and _template uses _build when defined
        rgb_logits, _ = rgb_model(rgb_clip_holder, is_training=is_train_holder,
                          dropout_keep_prob=dropout_holder)
        rgb_logits_dropout = tf.nn.dropout(rgb_logits, dropout_holder)
        # To change 400 classes to the ucf101 or hdmb classes
        rgb_fc_out = tf.layers.dense(
            rgb_logits_dropout, _CLASS_NUM[dataset], use_bias=True)
    
    with tf.variable_scope(_SCOPE['flow']):
        # insert i3d model
        flow_model = i3d.InceptionI3d(
            400, spatial_squeeze=True, final_endpoint='Logits')
        # the line below outputs the final results with logits
        # __call__ uses _template, and _template uses _build when defined
        flow_logits, _ = flow_model(flow_clip_holder, is_training=is_train_holder,
                          dropout_keep_prob=dropout_holder)
        flow_logits_dropout = tf.nn.dropout(flow_logits, dropout_holder)
        # To change 400 classes to the ucf101 or hdmb classes
        flow_fc_out = tf.layers.dense(
            flow_logits_dropout, _CLASS_NUM[dataset], use_bias=True)


    mixed_fc_out = _MIX_WEIGHT_OF_RGB * rgb_fc_out + _MIX_WEIGHT_OF_FLOW * flow_fc_out
        
    rgb_softmax_op = tf.nn.softmax(rgb_fc_out)
    flow_softmax_op = tf.nn.softmax(flow_fc_out)
    mixed_softmax_op = tf.nn.softmax(mixed_fc_out)
        
    rgb_in_top_1_op = tf.nn.in_top_k(rgb_softmax_op, rgb_label_holder, 1)
    flow_in_top_1_op = tf.nn.in_top_k(flow_softmax_op, flow_label_holder, 1)
    mixed_in_top_1_op = tf.nn.in_top_k(mixed_softmax_op, rgb_label_holder, 1)
    # Loss calculation, including L2-norm
    variable_map = {}
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        if tmp[0] == _SCOPE['rgb']:
            variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=variable_map)
    variable_map = {}
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        if tmp[0] == _SCOPE['flow']:
            variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=variable_map, reshape=True)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    rgb_saver.restore(sess, _CHECKPOINT_PATHS_RGB[int(split)-1])
    flow_saver.restore(sess, _CHECKPOINT_PATHS_FLOW[int(split)-1])
    
    # Specific Hyperparams
    # steps for training: the number of steps on batch per epoch
    
    print('----Here we start!----')
    print('Output wirtes to ' + log_dir)
    # logging.info('----Here we start!----')
    # for one epoch
    rgb_true_count = 0
    flow_true_count = 0
    mixed_true_count = 0
    # for 20 batches
    sess.run(test_init_op)
    
    rgb_outs = []
    flow_outs = []
    labels = []
    
    for i in range(rgb_data.size):
        rgb_in_top_1, flow_in_top_1, mixed_in_top_1, rgb_out, flow_out, label = sess.run([rgb_in_top_1_op, flow_in_top_1_op, mixed_in_top_1_op, rgb_fc_out, flow_fc_out, rgb_label_holder],
                                           feed_dict={dropout_holder: 1,
                                                      is_train_holder: False})
        
        
        rgb_true_count += np.sum(rgb_in_top_1)
        flow_true_count += np.sum(flow_in_top_1)
        mixed_true_count += np.sum(mixed_in_top_1)
        
        rgb_outs.append(rgb_out[0])
        flow_outs.append(flow_out[0])
        labels.append(label[0])
        
        print('rgb: %7d   flow: %7d   mixed:%7d   total: %7d/%d' %(rgb_true_count, flow_true_count, mixed_true_count, i, rgb_data.size))
        logging.info('rgb: %7d   flow: %7d   mixed:%7d' %(rgb_true_count, flow_true_count, mixed_true_count))
        
    rgb_accuracy = rgb_true_count / rgb_data.size
    flow_accuracy = flow_true_count / rgb_data.size
    mixed_accuracy = mixed_true_count / rgb_data.size
    
    rgb_outs = np.asarray(rgb_outs)
    flow_outs = np.asarray(flow_outs)
    labels = np.asarray(labels)
    
    result_data = pd.concat([pd.DataFrame(rgb_outs, columns=['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8']),
                             pd.DataFrame(flow_outs, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']), 
                             pd.DataFrame(labels, columns=['label'])], axis=1)
    
    result_data.to_csv(os.path.join(_LOG_ROOT,'test','result-%d.csv' % (split)))
    
    print('Accuracy:  rgb-%.3f   flow-%.3f   mixed-%.3f' %(rgb_accuracy, flow_accuracy, mixed_accuracy))
    logging.info('Accuracy:  rgb-%.3f   flow-%.3f   mixed-%.3f' %(rgb_accuracy, flow_accuracy, mixed_accuracy))
    
    sess.close()


if __name__ == '__main__':
    description = 'Finetune I3D model on other datasets (such as UCF101 and \
        HMDB51)'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('mode', type=str, help="type of data, e.g., rgb")
    p.add_argument('split', type=int, help="split of data, e.g., 1")
    main(**vars(p.parse_args()))
