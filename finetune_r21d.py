#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
import argparse
import time
import logging

import numpy as np
import tensorflow as tf

import pickle

import r2plus1d
from lib.action_dataset import Action_Dataset
from lib.action_dataset import split_data

_BATCH_SIZE = 3
_CLIP_SIZE = 16
# How many frames are used for each video in testing phase

_FRAME_SIZE = 112
_PREFETCH_BUFFER_SIZE = 30
_NUM_PARALLEL_CALLS = 2
_WEIGHT_OF_LOSS_WEIGHT = 7e-7
_MOMENTUM = 0.9
_DROPOUT = 0.36
_LOG_ROOT = 'output_r21d_16_run-04'

_CHECKPOINT_PATHS = {
    'rgb': './data/checkpoints/rgb_scratch/model.ckpt',
    'flow': './data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': './data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': './data/checkpoints/flow_imagenet/model.ckpt',
    'rgb_kin600': './data/checkpoints/rgb_kin600/model.ckpt',
    'r21d_rgb': './data/checkpoints/r21d/r2.5d_d34_l32_ft_sports1m.pkl',
    'r21d_flow': './data/checkpoints/r21d/r2.5d_d34_l32_ft_sports1m_optical_flow.pkl'
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
    'clipped_data': 8
}


def _get_data_label_from_info(train_info_tensor, name, mode):
    """ Wrapper for `tf.py_func`, get video clip and label from info list."""
    with tf.device('/cpu:0'):
        clip_holder, label_holder = tf.py_func(
            process_video, [train_info_tensor, name, mode], [tf.float32, tf.int32])
    return clip_holder, label_holder


def process_video(data_info, name, mode, is_training=True):
    """ Get video clip and label from data info list."""
    data = Action_Dataset(name, mode, [data_info])
    if is_training:
        clip_seq, label_seq = data.next_batch(1, _CLIP_SIZE, shuffle=True, data_augment=True, frame_size=_FRAME_SIZE)
    else:
        clip_seq, label_seq = data.next_batch(
            1, _CLIP_SIZE, shuffle=False, data_augment=False, frame_size=_FRAME_SIZE)
    clip_seq = 2*(clip_seq/255) - 1
    clip_seq = np.array(clip_seq, dtype='float32')
    return clip_seq, label_seq


def main(dataset='clipped_data', mode='rgb', split=1, investigate=0):
    assert mode in ['rgb', 'flow'], 'Only RGB data and flow data is supported'
    log_dir = os.path.join(_LOG_ROOT, 'finetune-%s-%s-%d' %
                           (dataset, mode, split))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'log.txt'),
                        filemode='w', format='%(message)s')

    ##  Data Preload  ###
    train_info, test_info = split_data(
            os.path.join('./data', dataset, mode+'.csv'),
            os.path.join('./data', dataset, 'testlist%02d' % split+'.txt'))
    train_data = Action_Dataset(dataset, mode, train_info)
    test_data = Action_Dataset(dataset, mode, test_info)
    
    
    
    num_train_sample = len(train_info)
    train_info_tensor = tf.constant(train_info)
    test_info_tensor = tf.constant(test_info)

    train_info_dataset = tf.data.Dataset.from_tensor_slices((train_info_tensor))
    train_info_dataset = train_info_dataset.shuffle(buffer_size=num_train_sample)
    train_dataset = train_info_dataset.map(lambda x: _get_data_label_from_info(
            x, dataset, mode), num_parallel_calls=_NUM_PARALLEL_CALLS)
    train_dataset = train_dataset.repeat().batch(_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    test_info_dataset = tf.data.Dataset.from_tensor_slices((test_info_tensor))
    test_dataset = test_info_dataset.map(lambda x: _get_data_label_from_info(
            x, dataset, mode), num_parallel_calls=_NUM_PARALLEL_CALLS)
    test_dataset = test_dataset.batch(1).repeat()
    test_dataset = test_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    # iterator = dataset.make_one_shot_iterator()
    # clip_holder, label_holder = iterator.get_next()
    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    clip_holder, label_holder = iterator.get_next()
    clip_holder = tf.squeeze(clip_holder,  [1])
    label_holder = tf.squeeze(label_holder, [1])
    clip_holder.set_shape(
        [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL[mode]])
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)

    # inference module
    # Inference Module
    model = r2plus1d.R2Plus1D()
        # the line below outputs the final results with logits
        # __call__ uses _template, and _template uses _build when defined
    logits = model(clip_holder, is_training=is_train_holder)
    logits_dropout = tf.nn.dropout(logits, dropout_holder)
        # To change 400 classes to the ucf101 or hdmb classes
    fc_out = tf.layers.dense(logits_dropout, _CLASS_NUM[dataset], use_bias=True)
    #print(fc_out.shape)
        # compute the top-k results for the whole batch size
    is_in_top_1_op = tf.nn.in_top_k(fc_out, label_holder, 1)

    # Loss calculation, including L2-norm
    variable_map = {}
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        variable_map[variable.name.replace('R2Plus1D/','')
                            .replace('/','_')
                            .replace(':0','')
                            .replace('gamma','s')
                            .replace('beta','b')
                            .replace('moving_mean','rm')
                            .replace('moving_variance','riv')
                            .replace('_1_conv', '_conv_1')
                            .replace('_2_conv', '_conv_2')
                            .replace('conv_2_1', '2_conv_1')
                            .replace('_1_spatbn_m', '_spatbn_1_m')
                            .replace('_2_spatbn_m', '_spatbn_2_m')
                            .replace('comp_3_shortcut_projection', 'shortcut_projection_3')
                            .replace('comp_7_shortcut_projection', 'shortcut_projection_7')
                            .replace('comp_13_shortcut_projection', 'shortcut_projection_13')
                            .replace('kernel', 'w')
                            .replace('bias', 'b')] = variable
        if tmp[-1] == 'w:0' or tmp[-1] == 'kernel:0':
            weight_l2 = tf.nn.l2_loss(variable)
            tf.add_to_collection('weight_l2', weight_l2)

    loss_weight = tf.add_n(tf.get_collection('weight_l2'), 'loss_weight')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_holder, logits=fc_out))
    total_loss = loss + _WEIGHT_OF_LOSS_WEIGHT * loss_weight
    #tf.summary.scalar('loss', loss)
    #tf.summary.scalar('loss_weight', loss_weight)
    #tf.summary.scalar('total_loss', total_loss)

    # Import Pre-trainned model
    #saver = tf.train.Saver(var_list=variable_map, reshape=True)
    saver2 = tf.train.Saver(max_to_keep=9999)
    # Specific Hyperparams
    # steps for training: the number of steps on batch per epoch
    per_epoch_step = int(np.ceil(train_data.size/_BATCH_SIZE))
    # global step constant
    if mode == 'flow':
        _GLOBAL_EPOCH = 45
        boundaries = [20000, 30000, 35000, 40000]
        values = [1e-3, 8e-4, 5e-4, 3e-4, 1e-4]
    else:
        _GLOBAL_EPOCH = 20
        boundaries = [900, 1500, 2000, 2500, 3000 ]
        values = [1e-3, 8e-4, 5e-4, 3e-4, 1e-4, 5e-5]
    global_step = _GLOBAL_EPOCH * per_epoch_step
    # global step counting
    global_index = tf.Variable(0, trainable=False)

    # Set learning rate schedule by hand, also you can use an auto way
    learning_rate = tf.train.piecewise_constant(
        global_index, boundaries, values)
    
    #tf.summary.scalar('learning_rate', learning_rate)

    # Optimizer set-up
    # FOR BATCH norm, we then use this updata_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               _MOMENTUM).minimize(total_loss, global_step=global_index)
    '''
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step=global_index)
    '''
    sess = tf.Session()
    #merged_summary = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op)
    
    # Load pretrained weight
    if mode == 'rgb':
        weights_file = _CHECKPOINT_PATHS['r21d_rgb']
    else:
        weights_file = _CHECKPOINT_PATHS['r21d_flow']
        
    with open(weights_file, 'rb') as fopen:
        blobs = pickle.load(fopen, encoding='latin-1')['blobs']
    
    print("len of blobs %d" % (len(blobs)))
    
    for k, v in sorted(blobs.items()):
        if k in variable_map:
            print('loading -- %s' % (k))
            if len(v.shape) == 2:
                sess.run(tf.assign(variable_map[k], tf.transpose(v)))
            elif len(v.shape) == 5:
                sess.run(tf.assign(variable_map[k], tf.transpose(v, perm=[2,3,4,1,0])))
            else:
                sess.run(tf.assign(variable_map[k], v))
    #saver.restore(sess, _CHECKPOINT_PATHS[train_data.mode+'_imagenet'])

    print('----Here we start!----')
    print('Output wirtes to ' + log_dir)
    # logging.info('----Here we start!----')
    step = 0
    
    true_count = 0
    epoch_completed = 0
    
    start_time = time.time()
    
    while step <= global_step:
        step += 1
        #start_time = time.time()
        _, is_in_top_1 = sess.run(
            [optimizer, is_in_top_1_op],
            feed_dict={dropout_holder: _DROPOUT, is_train_holder: True})
        #duration = time.time() - start_time
        if (investigate == 1) or (epoch_completed == _GLOBAL_EPOCH-1):
            tmp = np.sum(is_in_top_1)
            true_count += tmp
        
        #train_writer.add_summary(summary, step)
        
        if step % per_epoch_step == 0:
            epoch_completed += 1
            if (investigate == 1) or (epoch_completed == _GLOBAL_EPOCH):
                train_accuracy = true_count / (per_epoch_step * _BATCH_SIZE)
                true_count = 0
            
                sess.run(test_init_op)
                true_count = 0
                # start test process
                for i in range(test_data.size):
                    # print(i,true_count)
                    is_in_top_1 = sess.run(is_in_top_1_op,
                                           feed_dict={dropout_holder: 1,
                                                      is_train_holder: False})
                    true_count += np.sum(is_in_top_1)
                test_accuracy = true_count / test_data.size
                true_count = 0
                # to ensure every test procedure has the same test size
                test_data.index_in_epoch = 0
                print('Epoch%d - train: %.3f   test: %.3f   time: %d' %(epoch_completed, train_accuracy, test_accuracy, time.time() - start_time))
                logging.info('Epoch%d,train,%.3f,test,%.3f   time: %d' %(epoch_completed, train_accuracy, test_accuracy, time.time() - start_time))
                # saving the best params in test set
                saver2.save(sess, os.path.join(log_dir, test_data.name+'_'+train_data.mode), epoch_completed)
                sess.run(train_init_op)
            else:
                print('Epoch%d - time: %d' %(epoch_completed, time.time() - start_time))
                logging.info('Epoch%d time: %d' %(epoch_completed, time.time() - start_time))
            start_time = time.time()
    #train_writer.close()
    sess.close()


if __name__ == '__main__':
    description = 'Finetune I3D model on other datasets (such as UCF101 and \
        HMDB51)'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('mode', type=str, help="type of data, e.g., rgb")
    p.add_argument('split', type=int, help="split of data, e.g., 1")
    p.add_argument('investigate', type=int, help="0-no 1-yes")
    main(**vars(p.parse_args()))
