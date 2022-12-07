##############################################
## Author: junha park, github.com/hahajjjun ##
##############################################
# %% Imports
import os
import argparse
import numpy as np
import pandas as pd
import monai
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
import albumentations as A
import matplotlib.pyplot as plt
import cv2
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from monai.transforms import *
from monai.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

from onn import ONN

from mlflow import log_metric, log_param, log_artifacts
import mlflow.pytorch


# %%
random_seed = 40
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
monai.utils.set_determinism(seed=random_seed, additional_settings=None) 

# %% Models
class CustomModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(model_name='efficientnet_b1')
        self.dimred = nn.Linear(1000, 64)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dimred(x)
        return x

# %% Trainer
class CustomTrainer:

    def __init__(self):
        with open('./pool.json', 'r') as f:
            path_list = json.load(f)['PATH']
        random.shuffle(path_list)
        dataset_size = len(path_list)
        train_size = int(dataset_size * 0.8)
        validation_size = int(dataset_size * 0.2)

        train_paths = path_list[:train_size]
        valid_paths = path_list[train_size:train_size+validation_size]

        self.train_files = train_paths
        self.valid_files = valid_paths
        
        self.feature_extractor = CustomModel()
        pretrained_dict = torch.load('feature_extractor.pth')
        model_dict = self.feature_extractor.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        self.feature_extractor.load_state_dict(pretrained_dict)

        for name, param in self.feature_extractor.named_parameters():
            if name in pretrained_dict.keys():
                param.requires_grad = False

        self.feature_extractor.load_state_dict(model_dict)
        self.onn_network = ONN(features_size=64, max_num_hidden_layers=2, qtd_neuron_per_hidden_layer=12, n_classes=7)

    def preprocessor(self, img_path):
        encoder = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
        }
        img = np.array(Image.open(img_path))
        if img.shape[2] >3:
            img = img[:,:,:3]

        normalize = A.Compose(
            A.Sequential([
                A.Normalize(),
                A.Resize(224, 224)
            ])
        )

        img = normalize(image=img)['image']
        img = np.transpose(img, axes=(2,0,1))
        img = torch.tensor(img)
        label = img_path.split('/')[-2]
        label = np.array([encoder[label]])
        img = self.feature_extractor(torch.tensor((img[None,:,:,:]), dtype=torch.float32)).detach().cpu().numpy()
        return img, label
        
    def online_learn(self):
        
        decoder = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        X_test = np.stack([self.preprocessor(self.valid_files[i])[0] for i in range(len(self.valid_files))], axis=0)
        y_test = np.stack([self.preprocessor(self.valid_files[i])[1] for i in range(len(self.valid_files))], axis=0)

        X_test = X_test.squeeze()
        y_test = y_test.squeeze()
        
        for i in range(len(self.train_files)):
            X_train, y_train = self.preprocessor(self.train_files[i])
            self.onn_network.partial_fit(X_train, y_train)
            print(f"Image {i+1}: {decoder[int(y_train[0])]}, {self.train_files[i]}")
            predictions = self.onn_network.predict(X_test)
            print("Online Accuracy: {}".format(accuracy_score(y_test, predictions)))
        
        print(y_test)
        print(predictions)

    
# %%
if __name__ == "__main__":
    trainer = CustomTrainer()
    trainer.online_learn()