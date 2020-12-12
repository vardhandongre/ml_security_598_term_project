import time
import os
import json
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
import pickle


class DFDCDataset:
    def __init__(self, path='dfdc', split='train'):
        self.path = path
        self.videos = os.listdir(path)
        self.metadata = json.load(open('../baselines/deepfake-detection-challenge/train_sample_videos/metadata.json'))
        self.frames = []
        self.videos = self.videos[:355]
        for video in tqdm(self.videos):
            frame_list = []
            frames = os.listdir(os.path.join(self.path, video))
            for frame in frames:
                img = os.path.join(self.path, video, frame)
                img = self.load_image(img)
                img = img.reshape((1080, 1920, 3))
                frame_list.append(img)
            self.frames.append(np.stack(frame_list))


    def load_image(self, infilename):
        # img = Image.open( infilename )
        # img.load()
        img = cv2.imread(infilename)
        scale_percent = 20 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        data = np.asarray( img, dtype="float32" )
        return data

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset
        """
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.metadata[f"{video}.mp4"]["label"]
        if label == 'FAKE':
            label = 1
        else:
            label = 0
        frames = self.frames[idx]
        return frames, label



if __name__ == '__main__':
    dataset = DFDCDataset()
    pickle.dump(dataset, open('dfdc_dataset.pkl', 'wb'))
    exit()
    
    # for data in dataset:
    #     frames = data[0]
    #     print(frames.shape)

    dataset = pickle.load(open('dfdc_dataset.pkl', 'rb'))
    new_frames = []
    for frames in dataset.frames:
        for frame in frames:
            print(frame.shape)
            exit()

    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # for item in train_loader:
    #     print(item)
    #     exit()
