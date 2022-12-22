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
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from monai.transforms import *
from monai.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

from onn import ONN
from sklearn.neural_network import MLPClassifier

from mlflow import log_metric, log_param, log_artifacts
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

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
client = MlflowClient()

# %% Models
class CustomModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(model_name='efficientnet_b1', num_classes=64, pretrained=True)
        self.dimred = nn.Linear(64, 7)
        #self.fc = nn.Linear(64,7)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dimred(x)
        #x = self.fc(x)
        return x

class CustomEncoderModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(model_name='efficientnet_b1', num_classes=64)
    
    def forward(self, x):
        x = self.model(x)
        return x

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=7, smoothing=0.1, weight=None, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CustomDataset(Dataset):
        def __init__(self, pths, mode='train', prob = {'Gaussian':0.2, 'Brightness':0.2, 'Colorjitter':0.2, 'Affine':0.2, 'Elastic':0.2}):
            
            self.mode = mode
            self.paths = pths
            self.verbosity = True

        def __len__(self):

            return len(self.paths)


        def __getitem__(self, index):
            image_path = self.paths[index]
            encoder = {
                'angry': 0,
                'disgust': 1,
                'fear': 2,
                'happy': 3,
                'sad': 4,
                'surprise': 5,
                'neutral': 6
            }
            img = np.array(Image.open(image_path))
            if len(img.shape)==2:
                img = np.stack([img, img, img], axis=2)
            if img.shape[2] >3:
                img = img[:,:,:3]

            self.train_transform = A.Compose(
                A.Sequential([
                    A.OneOf([
                        A.Sequential([
                            A.GaussianBlur(p=0.2),
                            A.RandomBrightnessContrast(p=0.2),
                            A.ColorJitter(p=0.2),
                        ])
                    ]),
                    A.Normalize(),
                    A.Resize(224,224)
                ])
            )

            self.val_transform = A.Compose(
                A.Sequential([
                    A.Normalize(),
                    A.Resize(224,224)
                ])
            )

            if self.mode == 'train':
                img = self.train_transform(image=img)['image']
            elif (self.mode == 'valid')|(self.mode =='test'):
                img = self.val_transform(image=img)['image']
            img = np.transpose(img, axes=(2,0,1))
            img = torch.tensor(img)
            if (self.mode == 'train')|(self.mode == 'valid'):
                label = image_path.split('/')[-2]
                label = torch.tensor([encoder[label]])
                return img, label
            elif self.mode == 'test':
                return img
# %% Trainer
class CustomTrainer:

    def __init__(self):
        path_list = []
        for subdir in glob('../data/online_raw/*'):
            path_list.extend(sorted(glob(subdir+'/*')))
        random.shuffle(path_list)
        dataset_size = len(path_list)
        train_size = int(dataset_size * 0.8)
        validation_size = int(dataset_size * 0.2)

        train_paths = path_list[:train_size]
        valid_paths = path_list[train_size:train_size+validation_size]

        self.train_files = train_paths
        self.valid_files = valid_paths

        self.train_dataset = CustomDataset(self.train_files, 'train')
        self.valid_dataset = CustomDataset(self.valid_files, 'valid')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=8, shuffle=False)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=1, shuffle=False)

        self.train_partial_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.fullmodel = CustomModel().to(self.device)
        self.encoder = CustomEncoderModel()
        self.onn_network = MLPClassifier()
        self.optimizer = torch.optim.Adam(self.fullmodel.parameters(), lr = 1e-3, weight_decay = 1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'train_f1': [],
            'valid_f1': [],
            'train_acc': [],
            'valid_acc': []
        }
        self.best = 0

    def score_function(self, real, pred, mode='f1'):
        if mode == 'f1':
            score = f1_score(real, pred, average="macro")
        if mode == 'acc':
            score = accuracy_score(real, pred)
        return score

    def train_full(self):
        self.train_loss = 0
        self.train_pred=[]
        self.train_y=[]
        self.fullmodel.train()
        for _, (x,y) in tqdm(enumerate(self.train_dataloader)):
            self.optimizer.zero_grad()
            x = x.to(self.device, dtype = torch.float)
            y = y.to(self.device, dtype = torch.long).squeeze()
            y_pred = self.fullmodel(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.item()/len(self.train_dataloader)
            self.train_pred += y_pred.argmax(1).detach().cpu().numpy().tolist()
            self.train_y += y.detach().cpu().numpy().tolist()
            self.train_f1 = self.score_function(self.train_y, self.train_pred, mode='f1')
            self.train_acc = self.score_function(self.train_y, self.train_pred, mode='acc')
    
    def valid_full(self):
        self.valid_loss = 0
        self.valid_pred=[]
        self.valid_y=[]
        self.fullmodel.eval()
        for _, (x,y) in tqdm(enumerate(self.valid_dataloader)):
            self.optimizer.zero_grad()
            x = x.to(self.device, dtype = torch.float)
            y = y.to(self.device, dtype = torch.long)[:,0]
            y_pred = self.fullmodel(x)
            loss = self.criterion(y_pred, y)
            self.valid_loss += loss.item()/len(self.valid_dataloader)
            self.valid_pred += y_pred.argmax(1).detach().cpu().numpy().tolist()
            self.valid_y += y.detach().cpu().numpy().tolist()
            self.valid_f1 = self.score_function(self.valid_y, self.valid_pred, mode='f1')
            self.valid_acc = self.score_function(self.valid_y, self.valid_pred, mode='acc')
    
    def run_full(self):
        print("Start Quick Pretraining")
        for self.epoch in range(50):
            self.train_full()
            self.valid_full()
            
            self.history['train_loss'].append(self.train_loss)
            self.history['valid_loss'].append(self.valid_loss)
            self.history['train_f1'].append(self.train_f1)
            self.history['valid_f1'].append(self.valid_f1)
            self.history['train_acc'].append(self.train_acc)
            self.history['valid_acc'].append(self.valid_acc)

            if self.valid_acc > self.best:
                self.best = self.valid_acc
                torch.save(self.fullmodel.state_dict(),'best.pth')
            self.log_image()
    
    def train_partial(self, x, y):
        x_embed = self.encoder(x).detach().cpu().numpy()
        self.onn_network = self.onn_network.partial_fit(x_embed, y, np.unique(self.y_test))
    
    def valid_partial(self):
        predictions = self.onn_network.predict(self.x_test)
        acc = accuracy_score(self.y_test, predictions)
        print(acc)
        mlflow.log_metric('accuracy', acc)
        #print(self.y_test.squeeze())
        #print(predictions)

    def run_partial_unit(self, x, y):
        self.train_partial(x,y)
        self.valid_partial()

    def run_partial(self):
        pretrained_dict = torch.load('best.pth')
        model_dict = self.encoder.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.encoder.load_state_dict(model_dict)
        self.encoder = self.encoder.to(self.device)
        self.x_test = []
        self.y_test = []
        print("Generating test set")
        for _, (x,y) in enumerate(self.valid_dataloader):
            self.x_test.append(self.encoder(x.to(self.device)).detach().cpu().numpy())
            self.y_test.append(y.numpy())
        self.x_test = np.stack(self.x_test, axis=0).squeeze(1)
        self.y_test = np.stack(self.y_test, axis=0).squeeze(1)

        np.save('feature.npy', self.x_test)
        np.save('label.npy', self.y_test)

        print("Start OL")
        with mlflow.start_run() as run:
            for i, (x,y) in enumerate(self.train_partial_dataloader):
                mlflow.sklearn.log_model(self.onn_network, "model")  # logging scripted model
                if i == 0:
                    model_path = mlflow.get_artifact_uri("model")
                    print(f"Model saved @ {model_path}")
                    with open('recent_model_uri.log', 'w') as f:
                        f.write(model_path)
                x = x.to(self.device)
                y = y.numpy()
                print('Iteration', i, end=' ')
                self.run_partial_unit(x,y)

    def inference_partial(self, path, model_uri):
        pretrained_dict = torch.load('best.pth')
        model_dict = self.encoder.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.encoder.load_state_dict(model_dict)
        self.encoder = self.encoder.to(self.device)
        img = torch.tensor(CustomDataset([path], mode='test').__getitem__(0)[None,...], device=self.device)
        x_embed = self.encoder(img).detach().cpu().numpy()
        self.onn_network = mlflow.sklearn.load_model(model_uri=model_uri)
        y = self.onn_network.predict(x_embed)
        prob = self.onn_network.predict_proba(x_embed)
        return y, prob

    def log_image(self):
        # from nnU-Net snippet
        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()

        x_values = list(range(self.epoch + 1))
        ax.plot(x_values, self.history['train_loss'], color='b', ls='-', label="loss_tr")
        ax.plot(x_values, self.history['valid_loss'], color='b', ls='--', label="loss_val")
        ax2.plot(x_values, self.history['train_f1'], color='g', ls='-', label="f1_tr")
        ax2.plot(x_values, self.history['valid_f1'], color='g', ls='--', label="f1_val")
        ax2.plot(x_values, self.history['train_acc'], color='r', ls='-', label="acc_tr")
        ax2.plot(x_values, self.history['valid_acc'], color='r', ls='--', label="acc_val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("evaluation metric")
        ax.legend()
        ax2.legend(loc=9)

        fig.savefig("progress.png")
        plt.close()

    '''
    client.log_batch(mlflow.active_run().info.run_id, metrics=[Metric(key="ground truth", value=val, step=0) for val in y_test])
        client.log_batch(mlflow.active_run().info.run_id, metrics=[Metric(key="prediction", value=val, step=0) for val in predictions])
    '''
        

    
# %%
if __name__ == "__main__":
    trainer = CustomTrainer()
    #trainer.run_full()
    #trainer.run_partial()
    with open('recent_model_uri.log', 'r') as f:
        model_uri = f.readline()
    img_path = '/home/jhpark/InfantinO/modeling/src/data/online_raw/disgust/KakaoTalk_20221222_150935369.jpg'
    print("Fetch model from", model_uri)
    print("Inference on", img_path)
    prediction, uncertainty = trainer.inference_partial(path=img_path, model_uri=model_uri)
    print(list(prediction), list(uncertainty[0,]))