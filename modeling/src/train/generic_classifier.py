##############################################
## Author: junha park, github.com/hahajjjun ##
##############################################
# %% Imports
import wandb
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
import mlflow.pytorch

from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from monai.transforms import *
from monai.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score



# %%
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
monai.utils.set_determinism(seed=random_seed, additional_settings=None)
#  %% Dataset
class CustomDataset(Dataset):

    def __init__(self, img_pths, label_pths, mode='train', prob = {'Gaussian':0.2, 'Brightness':0.2, 'Colorjitter':0.2, 'Affine':0.2, 'Elastic':0.2}):
        
        self.mode = mode
        self.paths = img_pths
        self.labels = np.load(label_pths)
        self.verbosity = True
        
        self.train_transform = A.Compose(
            A.Sequential([
                A.OneOf([
                    A.Sequential([
                        A.GaussianBlur(p=prob['Gaussian']),
                        A.RandomBrightnessContrast(p=prob['Brightness']),
                        A.ColorJitter(p=prob['Colorjitter']),
                    ])
                ]),
                A.OneOf([
                    A.Sequential([
                        A.augmentations.geometric.Affine(scale=[0.8,1.2], translate_px=10, rotate=[-30,30], p=prob['Affine']),
                        A.augmentations.geometric.transforms.ElasticTransform(p=prob['Elastic'])
                    ])
                ]),

                A.HorizontalFlip(p=0.5),
                A.Normalize(),
            ])
        )

        self.val_transform = A.Compose(
            A.Sequential([
                A.Normalize(),
            ])
        )

    def __len__(self):

        return len(self.paths)


    def __getitem__(self, index):

        image_path = self.paths[index]
        img = np.array(np.load(image_path), dtype=np.float32)
        index = int(image_path.split('/')[-1].split('_')[-1].split('.')[0])
        label = self.labels[index]
        if img.shape[0] > 3:
            img = img[:3,:,:] #Use only 3 channels as input
            if self.verbosity:
                print(f'{image_path} input is modified since image has more than 3 channels')
                self.verbosity = False

        if self.mode == 'train':
            img = np.transpose(img, (1,2,0))
            img = self.train_transform(image=img)['image']
            img = A.resize(img, 224, 224)
            img = np.transpose(img, (2,0,1))

        else:
            img = np.transpose(img, (1,2,0))
            img = self.val_transform(image=img)['image']
            img = A.resize(img, 224, 224)
            img = np.transpose(img, (2,0,1))

        return img, label
# %% Models
class CustomModel(nn.Module):
    
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained = True)
        self.dimred = nn.Linear(1000, 64)
        self.head = nn.Linear(64,7)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dimred(x)
        x = self.head(x)
        return x
# %% Losses
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

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
# %% Trainer
class CustomTrainer:

    def __init__(self, args):

        self.root_dir = args.root
        self.label_pths = '/'.join(self.root_dir.split('/')[:-1])+'/train_lb.npy'
        self.args = args
        self.sub_dirs = os.listdir(self.root_dir)
        self.best = np.inf
        self.device = torch.device('cuda:0')
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'train_f1': [],
            'valid_f1': [],
            'train_acc': [],
            'valid_acc': []
        }

        path = self.root_dir+'/*.npy'
        path_list = sorted(glob(path))
        random.shuffle(path_list)
        dataset_size = len(path_list)
        train_size = int(dataset_size * 0.8)
        validation_size = int(dataset_size * 0.1)

        train_paths = path_list[:train_size]
        valid_paths = path_list[train_size:train_size+validation_size]
        test_paths = path_list[train_size+validation_size:]

        self.train_files = train_paths
        self.valid_files = valid_paths
        self.test_files = test_paths
        
        self.train_loader, self.valid_loader, self.test_loader = self.get_dataloader()
        self.model = CustomModel(self.args.model).to(self.device)
        
        self.optimizer_map = {
            'Adam': torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.decay),
            'AdamW': torch.optim.AdamW(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.decay),
            'SGD': torch.optim.SGD(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.decay),
        }

        self.optimizer = self.optimizer_map[self.args.optimizer]

        self.scheduler_map = {
            'StepLR':torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma = 0.1),
            'CosineAnnealWarmRestartsLR': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=self.args.lr/100)
        }

        self.scheduler = self.scheduler_map[self.args.scheduler]

        self.loss_map = {
            'CE': torch.nn.CrossEntropyLoss(),
            'Smooth': LabelSmoothingLoss(),
            'Focal': FocalLoss()
        }

        self.criterion = self.loss_map[self.args.loss]
    
    def get_dataloader(self):
        prob = {
            'Gaussian':self.args.gaussianprob,
            'Brightness':self.args.brightnessprob,
            'Colorjitter':self.args.colorjitterprob,
            'Affine':self.args.affineprob,
            'Elastic':self.args.elasticprob
        }
        train_dataset = CustomDataset(self.train_files, self.label_pths, mode='train', prob = prob)
        valid_dataset = CustomDataset(self.valid_files, self.label_pths, mode='valid')
        test_dataset = CustomDataset(self.test_files, self.label_pths, mode='test')
        train_dataloader = DataLoader(train_dataset, batch_size = self.args.batch_size, shuffle = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size = self.args.batch_size, shuffle = False)
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

        return train_dataloader, valid_dataloader, test_dataloader
    
    def train(self, dataloader):
        self.train_loss = 0
        self.train_pred=[]
        self.train_y=[]
        self.model.train()
        for _, (x,y) in tqdm(enumerate(dataloader)):
            self.optimizer.zero_grad()
            x = x.to(self.device, dtype = torch.float)
            y = y.to(self.device, dtype = torch.long)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.item()/len(dataloader)
            self.train_pred += y_pred.argmax(1).detach().cpu().numpy().tolist()
            self.train_y += y.detach().cpu().numpy().tolist()
            self.train_f1 = self.score_function(self.train_y, self.train_pred, mode='f1')
            self.train_acc = self.score_function(self.train_y, self.train_pred, mode='acc')
        self.scheduler.step()

    def valid(self, dataloader):
        self.valid_loss = 0
        self.valid_pred=[]
        self.valid_y=[]
        self.model.eval()
        for _, (x,y) in tqdm(enumerate(dataloader)):
            self.optimizer.zero_grad()
            x = x.to(self.device, dtype = torch.float)
            y = y.to(self.device, dtype = torch.long)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            self.valid_loss += loss.item()/len(dataloader)
            self.valid_pred += y_pred.argmax(1).detach().cpu().numpy().tolist()
            self.valid_y += y.detach().cpu().numpy().tolist()
            self.valid_f1 = self.score_function(self.valid_y, self.valid_pred, mode='f1')
            self.valid_acc = self.score_function(self.valid_y, self.valid_pred, mode='acc')

    def score_function(self, real, pred, mode='f1'):
        if mode == 'f1':
            score = f1_score(real, pred, average="macro")
        if mode == 'acc':
            score = accuracy_score(real, pred)
        return score

    def run(self):
        for self.epoch in range(self.args.epochs):
            self.train(self.train_loader)
            self.valid(self.valid_loader)
            
            self.history['train_loss'].append(self.train_loss)
            self.history['valid_loss'].append(self.valid_loss)
            self.history['train_f1'].append(self.train_f1)
            self.history['valid_f1'].append(self.valid_f1)
            self.history['train_acc'].append(self.train_acc)
            self.history['valid_acc'].append(self.valid_acc)

            #print(f'[{epoch+1}/{self.args.epochs}] [Train Loss : {self.train_loss}, Train F1 : {self.train_f1}, Train Acc : {self.train_acc}] [Valid Loss : {self.valid_loss}, Valid F1 : {self.valid_f1}, Valid Acc : {self.valid_acc}]')
            if self.valid_loss < self.best:
                self.best = self.valid_loss
                torch.save(self.model.state_dict(),f'{self.args.name}_{self.args.model}_{self.args.optimizer}_{self.args.lr}_{self.args.scheduler}.pth')
            self.log_image()
    def inference(self):
        self.model.load_state_dict(torch.load(f'{self.args.name}_{self.args.model}_{self.args.optimizer}_{self.args.lr}_{self.args.scheduler}.pth'))
        self.valid(self.test_loader)
        print(f'[Test F1 : {self.valid_f1}, Test Acc : {self.valid_acc}]')     

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

    
# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="infantino arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #* Setting
    parser.add_argument("-n", "--name", type=str, default="test", help="Name of the run.")
    parser.add_argument("--root", type=str, default="../data/processed/train", help="Dataset directory.")

    #* Augmentation
    parser.add_argument("--gaussianprob", type=float, default=0.2, help="Gaussian blur augmentation probability.")
    parser.add_argument("--brightnessprob", type=float, default=0.2, help="Random brightness augmentation probability.")
    parser.add_argument("--colorjitterprob", type=float, default=0.2, help="Colorjitter augmentation probability.")
    parser.add_argument("--affineprob", type=float, default=0.2, help="Affine augmentation probability.")
    parser.add_argument("--elasticprob", type=float, default=0.2, help="Elastic deformation augmentation probability.")

    #* Training
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Epoch number to run.")

    #* Model ---
    parser.add_argument("--model", type=str, default="efficientnet_b1", help="Timm named model. Check timm.list_models() for available model names.")    
    
    #* Dataloader
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")

    #* Loss
    parser.add_argument("--loss", type = str, default='CE', help="Loss types. ['CE', 'Smooth', 'Focal'] is available.")

    #* Optimizer & Scheduler
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=1e-3, help="Weight decay.")
    parser.add_argument("--optimizer", type=str, default='AdamW', help="Optimizer types. ['Adam', 'AdamW', 'SGD'] is available.")
    parser.add_argument("--scheduler", type=str, default='CosineAnnealWarmRestartsLR', help="Scheduler type. ['StepLR', 'CosineAnnealWarmRestartsLR'] is available.")
    args = parser.parse_args()

    print(args)


    # mlflow.pytorch.autolog() # -- 

    trainer = CustomTrainer(args)
    trainer.run()
    trainer.inference()