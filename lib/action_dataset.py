import numpy as np
from lib.video_3d import Video_3D
import os

class Action_Dataset:
    def __init__(self, name, mode, video_info):
        self.name = name
        self.mode = mode
        self.videos = [Video_3D(x, self.mode) for x in video_info]
        # the size is maybe the number of videos
        self.size = len(self.videos)
        self.epoch_completed = 0
        self.index_in_epoch = 0
        self.perm = np.arange(self.size)

        # Edited by Alex Hu
        np.random.shuffle(self.perm)

    def next_batch(self, batch_size, frame_num, shuffle=True, data_augment=True, frame_size=224):
        # used for counting the number of epoches,
        # end is current number of the total processed videos
        # index_in_epoch is the index of this epoch
        start = self.index_in_epoch
        end = start + batch_size
        self.index_in_epoch = end % self.size
        batch = []
        label = []
        #print(type(self.mode))
        #print(self.mode)
        if end >= self.size:
            self.epoch_completed += 1
            for i in range(start, self.size):
                # construct batch from start to the size
                batch.append(
                    self.videos[self.perm[i]].get_frames(frame_num, data_augment=data_augment, side_length=frame_size))
                label.append(self.videos[self.perm[i]].label)
            if shuffle:
                np.random.shuffle(self.perm)
            for i in range(0, self.index_in_epoch):
                batch.append(
                    self.videos[self.perm[i]].get_frames(frame_num, data_augment=data_augment, side_length=frame_size))
                label.append(self.videos[self.perm[i]].label)
        else:
            for i in range(start, end):
                batch.append(
                    self.videos[self.perm[i]].get_frames(frame_num, data_augment=data_augment, side_length=frame_size))
                label.append(self.videos[self.perm[i]].label)
        return np.stack(batch), np.stack(label)
    
    def get_element(self, batch_size, frame_num, frame_size=224):
        # used for counting the number of epoches,
        # end is current number of the total processed videos
        # index_in_epoch is the index of this epoch
        start = self.index_in_epoch
        end = start + batch_size
        self.index_in_epoch = end % self.size
        batch = []
        label = []
        #print(type(self.mode))
        #print(self.mode)
        if end >= self.size:
            self.epoch_completed += 1
            for i in range(start, self.size):
                # construct batch from start to the size
                batch.append(
                    self.videos[self.perm[i]].get_test_frames(frame_num, side_length=frame_size))
                label.append(self.videos[self.perm[i]].label)
            for i in range(0, self.index_in_epoch):
                batch.append(
                    self.videos[self.perm[i]].get_test_frames(frame_num, side_length=frame_size))
                label.append(self.videos[self.perm[i]].label)
        else:
            for i in range(start, end):
                batch.append(
                    self.videos[self.perm[i]].get_test_frames(frame_num, side_length=frame_size))
                label.append(self.videos[self.perm[i]].label)
        return np.stack(batch), np.stack(label)

def split_data(data_info, test_split):
    f1 = open(data_info, 'r')
    f2 = open(test_split, 'r')
    test = list()
    train_info = list()
    test_info = list()
    # extract the specific video name,and plus a document name behind,the result is such v_ApplyEyeMakeup_g01_c01
    for line in f2.readlines():
        test.append(line.split(os.path.sep)[1].split('.')[0])
    
    # if rgb's doc name is in testlist,then append in test_info,if not, append in train
    # for example, info is
    # v_ApplyEyeMakeup_g01_c01,/data4/zhouhao/dataset/ucf101/jpegs_256/v_ApplyEyeMakeup_g01_c01,165,0
    for line in f1.readlines():
        info = line.strip().split(',')
        if info[0] in test:
            test_info.append(info)
        else:
            train_info.append(info)
    f1.close()
    f2.close()
    return train_info, test_info

def get_each_frame_test_info(data_info, test_split, mode):
    f1 = open(data_info, 'r')
    f2 = open(test_split, 'r')
    test = list()
    test_info = list()
    # extract the specific video name,and plus a document name behind,the result is such v_ApplyEyeMakeup_g01_c01
    for line in f2.readlines():
        test.append(line.split(os.path.sep)[1].split('.')[0])
    
    # if rgb's doc name is in testlist,then append in test_info,if not, append in train
    # for example, info is
    # v_ApplyEyeMakeup_g01_c01,/data4/zhouhao/dataset/ucf101/jpegs_256/v_ApplyEyeMakeup_g01_c01,165,0
    for line in f1.readlines():
        info = line.strip().split(',')
        if info[0] in test:
            num_frame = int(info[2])
            for i in range(num_frame):
                temp_info = info.copy()
                temp_info.append(str(i+1))
                test_info.append(temp_info)
            
    f1.close()
    f2.close()
    return test_info