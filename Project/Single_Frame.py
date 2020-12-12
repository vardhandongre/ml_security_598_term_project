#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)


# In[2]:


# cd gdrive/My Drive/CS 598 GW/Project/


# In[2]:





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


# In[4]:


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

import pickle


# In[ ]:


###########################################         HELPER FUNCTIONS BELOW             #####################################################################


# In[27]:


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


# In[6]:


def get_meta_from_json(path):
    df = pd.read_json(path)
    df = df.T
    return df


# In[7]:


# Load Video Frames as array
def loadframe(args):
    (filename,augment) = args
    vid = []
    cap = cv2.VideoCapture()
    path = os.path.join(MAIN_DIR, TRAIN_DIR, filename)
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
    
    # Augment 
    mean = np.asarray([0.485, 0.456, 0.406],np.float32)
    std = np.asarray([0.229, 0.224, 0.225],np.float32)

    curr_w = 1920
    curr_h = 1080
    height = width = 500
    data = np.zeros((3,height,width),dtype=np.float32)
    nFrames = len(vid)
    frame_index = np.random.randint(nFrames)
    frame = vid_arr[frame_index]
    try:
        ### load file from HDF5
        nFrames = len(vid)
        frame_index = np.random.randint(nFrames)
        frame = vid_arr[frame_index]

        if(augment==True):
            ## RANDOM CROP - crop 70-100% of original size
            ## don't maintain aspect ratio
            if(np.random.randint(2)==0):
                resize_factor_w = 0.3*np.random.rand()+0.7
                resize_factor_h = 0.3*np.random.rand()+0.7
                w1 = int(curr_w*resize_factor_w)
                h1 = int(curr_h*resize_factor_h)
                w = np.random.randint(curr_w-w1)
                h = np.random.randint(curr_h-h1)
                frame = frame[h:(h+h1),w:(w+w1)]
            
            ## FLIP
            if(np.random.randint(2)==0):
                frame = cv2.flip(frame,1)

            frame = cv2.resize(frame,(width,height))
            frame = frame.astype(np.float32)

            ## Brightness +/- 15
            brightness = 30
            random_add = np.random.randint(brightness+1) - brightness/2.0
            frame += random_add
            frame[frame>255] = 255.0
            frame[frame<0] = 0.0

        else:
            # don't augment
            frame = cv2.resize(frame,(width,height))
            frame = frame.astype(np.float32)

        ## resnet model was trained on images with mean subtracted
        frame = frame/255.0
        frame = (frame - mean)/std
        frame = frame.transpose(2,0,1)
        data[:,:,:] = frame
    except:
        print("Exception: " + filename)
        data = np.array([])
    return data


# In[ ]:


###########################################         HELPER FUNCTIONS ABOVE             #####################################################################


# In[28]:


# DATASET

MAIN_DIR = 'data'
TRAIN_DIR = 'train_sample_videos'
train, test, df = data_splitter(MAIN_DIR,TRAIN_DIR)


# In[22]:


# HYPER-PARAMETERS
IMAGE_SIZE = 500
NUM_CLASSES = 2
batch_size = 10
lr = 0.0001
num_of_epochs = 3


# In[30]:


# SINGLE FRAME

model =  torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,NUM_CLASSES)

for param in model.parameters():
    param.requires_grad_(False)

for param in model.layer4[2].parameters():
    param.requires_grad_(True)
for param in model.fc.parameters():
    param.requires_grad_(True)

params = []

for param in model.layer4[2].parameters():
    params.append(param)
for param in model.fc.parameters():
    params.append(param)

model.cuda()

optimizer = optim.Adam(params,lr=lr)
criterion = nn.CrossEntropyLoss()

pool_threads = Pool(8,maxtasksperchild=200)

accs = []

for epoch in range(0,num_of_epochs):

    ###### TRAIN
    train_accu = []
    model.train()
    random_indices = np.random.permutation(len(train[0]))
    start_time = time.time()
    for i in range(0, len(train[0])-batch_size,batch_size):
        try:
          augment = True
          video_list = [(train[0][k],augment)
                        for k in random_indices[i:(batch_size+i)]]
          data = pool_threads.map(loadframe,video_list)

          next_batch = 0
          for video in data:
              if video.size==0: # there was an exception, skip this
                  next_batch = 1
          if(next_batch==1):
              continue

          x = np.asarray(data,dtype=np.float32)
          x = Variable(torch.FloatTensor(x)).cuda().contiguous()

          y = np.array(train[1])
          y = y[random_indices[i:(batch_size+i)]]
          y = torch.from_numpy(y).cuda()

          output = model(x)

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
        except ValueError:
            continue
    accs.append(accuracy_epoch)
    print(epoch, accuracy_epoch,time.time()-start_time)

torch.save(model,'single_frame.model')
pool_threads.close()
pool_threads.terminate()
pickle.dump(accs, open('results/accuracies_single_frame.pkl', 'wb'))

##### TEST
model.eval()
test_accu = []
random_indices = np.random.permutation(len(test[0]))
t1 = time.time()
for i in range(0,len(test[0])-batch_size,batch_size):
    augment = False
    video_list = [(test[0][k],augment) 
                    for k in random_indices[i:(batch_size+i)]]
    data = pool_threads.map(loadframe,video_list)

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

    output = model(x)

    prediction = output.data.max(1)[1]
    accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
    test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
print('Testing',accuracy_test,time.time()-t1)


# In[26]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[29]:


missing_data(df)


# In[19]:


x = np.array(train[1])

x[random_indices[0:(batch_size+0)]]


# In[23]:


[random_indices[80:(batch_size+80)]]


# In[ ]:




