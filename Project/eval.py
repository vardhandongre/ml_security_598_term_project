import shutil
import os.path
from os import path
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn import metrics

# from Sequential import data_splitter, loadSequence

import resnet_3d

from tqdm import trange

IMAGE_SIZE = 500
NUM_CLASSES = 2


model_type = 'Sequential'
assert model_type in ['Sequential', 'Frame']



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



def get_scores(pred, gold, thres=None):
	acc = metrics.accuracy_score(gold, pred)
	precision, recall, fscore, support = metrics.precision_recall_fscore_support(gold, pred, average='binary')
	return acc, precision , recall ,fscore, support


def get_auc(pred, gold):
	fpr, tpr, thresholds = metrics.roc_curve(gold, pred, pos_label=1)
	auc = metrics.auc(fpr, tpr)
	return fpr, tpr, thresholds, auc


def plot_roc_curve(tpr, fpr, roc_auc, fname='roc.png'):
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.legend(loc = 'lower right')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(fname)


# model =  resnet_3d.resnet101(sample_size=IMAGE_SIZE, sample_duration=16)


# dir = 'data/'
# pretrained = torch.load('3d_resnet.model')
# keys = [k for k,v in pretrained['state_dict'].items()]
# pretrained_state_dict = {k[7:]: v.cpu() for k, v in pretrained['state_dict'].items()}
# model.load_state_dict(pretrained_state_dict)


# model.load_state_dict(torch.load('3d_resnet.model'))

if model_type == 'Sequential':
	model = torch.load('backup/3d_resnet.model')

print("Loaded Model")

model.eval()


MAIN_DIR = 'data'
TRAIN_DIR = 'train_sample_videos'
train, test, df = data_splitter(MAIN_DIR,TRAIN_DIR)


# In[8]:


IMAGE_SIZE = 500
NUM_CLASSES = 2
# batch_size = 32
batch_size = 10
lr = 0.0001
num_of_epochs = 5

print("Creating Pool Threads")
pool_threads = Pool(8,maxtasksperchild=200)
print("Finished Creating Pool Threads")

test_accu = []
preds = []
labs = []
random_indices = np.random.permutation(len(test[0]))
t1 = time.time()
for i in trange(0,len(test[0])-batch_size,batch_size):
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
    preds.extend(list(prediction.cpu().numpy()))
    labs.extend(list(y.cpu().numpy()))
    test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)

print('Testing',accuracy_test,time.time()-t1)
acc, precision , recall ,fscore, support = get_scores(preds, labs)

print(f"Accuracy = {acc}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"F-Score = {fscore}")
print(f"Support = {support}")


fpr, tpr, thresholds, auc = get_auc(preds, labs)
print(f"AUC = {auc}")
if model_type == 'Sequential':
	plot_roc_curve(tpr, fpr, auc, fname='backup/roc_sequential.png')