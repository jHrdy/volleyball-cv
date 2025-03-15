import json
from pathlib import Path
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from losses import CEandDiceLoss
from utils import ClassItem, Classes
import numpy as np

classes = Classes([
        ClassItem('Honza', 0, 1, (23,187,76)),
        ClassItem('Jarda', 1, 1, (214,255,23)),
        ClassItem('Krystof', 2, 1, (89,134,179)),
        ClassItem('Kamilla', 3, 1, (250,50,83)),
        ClassItem('Jozef', 4, 1, (250,50,183)),
        ClassItem('Ball', 5, 1, (255,35,0)),
        ClassItem('background', 6, 1, (0,0,0)),
    ])

color_to_class = {c.rgb : c.index for c in classes}

def load_imgs(dir):
    imgs = []
    for img_pth in dir.iterdir():
        imgs.append(Image.open(img_pth))
    return imgs


def transform_mask(mask):
    mask = np.array(mask)
    class_matrix = np.zeros_like(mask)
    
    for i, row in enumerate(mask):
        for j, pixel in enumerate(row):
            class_matrix[i,j] = color_to_class[tuple(pixel)]
    
    return class_matrix

def train(dataset, model, config):
    dataloader = DataLoader(dataset, shuffle=True)
    
    ce_class_weights = torch.tensor([ci.ce_weight for ci in classes])

    criterion = CEandDiceLoss(ce_weight=0.5,dice_weight=0.5,ce_class_weights=ce_class_weights)    
    optimizer = Adam(lr=config['lr'], params=model.parameters())
    losses = []

    for ep in tqdm(range(config['num_epochs'])):
        epoch_loss = 0
        for batch in dataloader:
            x, y = batch[0], batch[1]
            y = transform_mask(y)
            
            ypred = model(x)
            loss = criterion(y,ypred)
            losses.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if not ep % 10:
            print(f'Epoch: {ep} | Loss: {epoch_loss/len(dataloader)}')

        
if __name__ == '__main__':
    with open('hyperparams.json', 'r') as f:
        config = json.load(f)

    masks_dir = Path(os.path.join(os.getcwd(),'frames','Data','SegmentationClass'))
    img_dir = Path(os.path.join(os.getcwd(),'frames','Photos'))
    
    imgs = load_imgs(img_dir)
    masks = load_imgs(masks_dir)

    
    