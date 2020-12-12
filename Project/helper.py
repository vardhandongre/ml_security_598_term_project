import shutil
import os.path
from os import path
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# Directories (add them)

# MAIN_DIR = ''
# TRAIN_DATA = ''
# TEST_DATA = ''

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
	return [X_train, y_train], [X_test, y_test]



def get_meta_from_json(path):
    df = pd.read_json(path)
    df = df.T
    return df


# Load Video Frames as array
def loadframe(args):
    (filename,augment) = args
    vid = []
    cap = cv2.VideoCapture()
    path = os.path.join(MAIN_DIR,filename)
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



def loadSequence(args):
    (filename,augment) = args
    vid = []
    cap = cv2.VideoCapture()
    path = os.path.join(MAIN_DIR,filename)
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