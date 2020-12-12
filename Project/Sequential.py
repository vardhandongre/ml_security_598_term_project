#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)


# In[4]:


# cd gdrive/My Drive/CS 598 GW/Project/


# In[ ]:


###########################################         DEPENDENCIES             #####################################################################


# In[3]:


import shutil
import os.path
from os import path
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

import h5py
import cv2

from multiprocessing import Pool


# In[ ]:


###########################################         HELPER FUNCTIONS BELOW             #####################################################################


# In[5]:


def data_splitter(MAIN_DIR, TRAIN_DATA):

	# list of training & test videos
	total_data = list(os.listdir(os.path.join(MAIN_DIR,TRAIN_DATA)))
	train_df = get_meta_from_json('data/train_sample_videos/metadata.json')
	train_df.reset_index(inplace=True)
	df = pd.DataFrame(columns = ['files', 'labels'])
	df.files = train_df['index']

	for i in range(len(df)):
		if train_df.iloc[i,1] == 'FAKE':
			df.iloc[i,1] = 1
		elif train_df.iloc[i,1] == 'REAL':
			df.iloc[i,1] = 0

	X = list(df.files)
	y = list(df.labels)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	return [X_train, y_train], [X_test, y_test] , df

def get_meta_from_json(path):
    df = pd.read_json(path)
    df = df.T
    return df


# In[ ]:





# In[6]:


def loadSequence(args):
    (filename,augment) = args
    vid = []
    cap = cv2.VideoCapture()
    path = os.path.join(MAIN_DIR,TRAIN_DIR,filename)
    cap.open(path)

    if not cap.isOpened():
        print("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_idx = 0

    while frame_idx < frame_count:
        ret, frame = cap.read()
        vid.append(frame)


        if not ret:
            print ("Failed to get the frame {}".format(frameId))
            continue
        frame_idx += 10
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    vid_arr = np.array(vid).reshape(len(vid), 1080, 1920, 3)
    
    mean = np.asarray([0.433, 0.4045, 0.3776],np.float32)
    std = np.asarray([0.1519876, 0.14855877, 0.156976],np.float32)

    curr_w = 1080
    curr_h = 1920
    height = width = 500
    num_of_frames = 16

    (filename,augment) = args

    data = np.zeros((3,num_of_frames,height,width),dtype=np.float32)

    try:
        nFrames = len(vid)
        frame_index = np.random.randint(nFrames - num_of_frames)
        video = vid_arr[frame_index:(frame_index + num_of_frames)]

        if(augment==True):
            ## RANDOM CROP - crop 70-100% of original size
            ## don't maintain aspect ratio
            resize_factor_w = 0.3*np.random.rand()+0.7
            resize_factor_h = 0.3*np.random.rand()+0.7
            w1 = int(curr_w*resize_factor_w)
            h1 = int(curr_h*resize_factor_h)
            w = np.random.randint(curr_w-w1)
            h = np.random.randint(curr_h-h1)
            random_crop = np.random.randint(2)

            ## Random Flip
            random_flip = np.random.randint(2)

            ## Brightness +/- 15
            brightness = 30
            random_add = np.random.randint(brightness+1) - brightness/2.0

            data = []
            for frame in video:
                if(random_crop):
                    frame = frame[h:(h+h1),w:(w+w1),:]
                if(random_flip):
                    frame = cv2.flip(frame,1)
                frame = cv2.resize(frame,(width,height))
                frame = frame.astype(np.float32)
                
                frame += random_add
                frame[frame>255] = 255.0
                frame[frame<0] = 0.0

                frame = frame/255.0
                frame = (frame - mean)/std
                data.append(frame)
            data = np.asarray(data)

        else:
            # don't augment
            data = []
            for frame in video:
                frame = cv2.resize(frame,(width,height))
                frame = frame.astype(np.float32)
                frame = frame/255.0
                frame = (frame - mean)/std
                data.append(frame)
            data = np.asarray(data)

        data = data.transpose(3,0,1,2)
    except:
        print("Exception: " + filename)
        data = np.array([])
    return data


# In[ ]:


###########################################         HELPER FUNCTIONS ABOVE             #####################################################################


# In[7]:


# DATASET

MAIN_DIR = 'data'
TRAIN_DIR = 'train_sample_videos'
train, test, df = data_splitter(MAIN_DIR,TRAIN_DIR)


# In[8]:


IMAGE_SIZE = 500
NUM_CLASSES = 2
# batch_size = 32
batch_size = 10
lr = 0.0001
num_of_epochs = 10


# In[ ]:


import resnet_3d
dir = 'data/'
# model =  resnet_3d.resnet50(sample_size=IMAGE_SIZE, sample_duration=16)
model =  resnet_3d.resnet101(sample_size=IMAGE_SIZE, sample_duration=16)
pretrained = torch.load(dir + 'resnet-101-kinetics.pth')
keys = [k for k,v in pretrained['state_dict'].items()]
pretrained_state_dict = {k[7:]: v.cpu() for k, v in pretrained['state_dict'].items()}
model.load_state_dict(pretrained_state_dict)
model.fc = nn.Linear(model.fc.weight.shape[1],NUM_CLASSES)

for param in model.parameters():
    param.requires_grad_(False)

# for param in model.conv1.parameters():
#     param.requires_grad_(True)
# for param in model.bn1.parameters():
#     param.requires_grad_(True)
# for param in model.layer1.parameters():
#     param.requires_grad_(True)
# for param in model.layer2.parameters():
#     param.requires_grad_(True)
# for param in model.layer3.parameters():
#     param.requires_grad_(True)
for param in model.layer4[0].parameters():
    param.requires_grad_(True)
for param in model.fc.parameters():
    param.requires_grad_(True)

params = []
# for param in model.conv1.parameters():
#     params.append(param)
# for param in model.bn1.parameters():
#     params.append(param)
# for param in model.layer1.parameters():
#     params.append(param)
# for param in model.layer2.parameters():
#     params.append(param)
# for param in model.layer3.parameters():
#     params.append(param)
for param in model.layer4[0].parameters():
    params.append(param)
for param in model.fc.parameters():
    params.append(param)


model.cuda()

optimizer = optim.Adam(params,lr=lr)

criterion = nn.CrossEntropyLoss()

pool_threads = Pool(8,maxtasksperchild=200)

for epoch in range(0,num_of_epochs):

    ###### TRAIN
    train_accu = []
    model.train()
    random_indices = np.random.permutation(len(train[0]))
    start_time = time.time()
    for i in range(0, len(train[0])-batch_size,batch_size):

        augment = True
        video_list = [(train[0][k],augment)
                       for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadSequence,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this
                next_batch = 1
        if(next_batch==1):
            continue

        x = np.asarray(data,dtype=np.float32)
        x = Variable(torch.FloatTensor(x),requires_grad=False).cuda().contiguous()
        y = np.array(train[1])
        y = y[random_indices[i:(batch_size+i)]]
        y = torch.from_numpy(y).cuda()

        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
        h = model.layer4[0](h)

        h = model.avgpool(h)

        h = h.view(h.size(0), -1)
        output = model.fc(h)

        # output = model(x)

        loss = criterion(output, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
        if(epoch==0):
            print(i,accuracy)
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch,time.time()-start_time)

    ##### TEST
    model.eval()
    test_accu = []
    random_indices = np.random.permutation(len(test[0]))
    t1 = time.time()
    for i in range(0,len(test[0])-batch_size,batch_size):
        augment = False
        video_list = [(test[0][k],augment) 
                        for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadSequence,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this batch
                next_batch = 1
        if(next_batch==1):
            continue

        x = np.asarray(data,dtype=np.float32)
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()
        y = np.array(test[1])

        y = y[random_indices[i:(batch_size+i)]]
        y = torch.from_numpy(y).cuda()

        # with torch.no_grad():
        #     output = model(x)
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4[0](h)
            # h = model.layer4[1](h)

            h = model.avgpool(h)

            h = h.view(h.size(0), -1)
            output = model.fc(h)

        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
        test_accu.append(accuracy)
        accuracy_test = np.mean(test_accu)

    print('Testing',accuracy_test,time.time()-t1)

torch.save(model,'3d_resnet.model')
pool_threads.close()
pool_threads.terminate()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




