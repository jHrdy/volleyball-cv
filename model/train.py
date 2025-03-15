import json
from pathlib import Path
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from losses import CEandDiceLoss
from utils import ClassItem, Classes, ImgDataset
import numpy as np
from model import SegmentationModel
import matplotlib.pyplot as plt
from torchvision import transforms

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
class_to_color = {c.index : c.rgb for c in classes}

def load_imgs(dir):
    imgs = []
    for img_pth in dir.iterdir():
        imgs.append((img_pth, Image.open(img_pth)))
    imgs = sorted(imgs, key=lambda kv: kv[0])
    return [i[1] for i in imgs]


def transform_mask(mask):
    mask = np.array(mask)
    class_matrix = np.zeros_like(mask)
    
    for i, row in enumerate(mask):
        for j, pixel in enumerate(row):
            class_matrix[i,j] = color_to_class[tuple(pixel)]
    return class_matrix.astype(np.uint8)

def transform_output(output):
    output = output.squeeze(dim=0)
    output = torch.argmax(output, dim=0).to(torch.int32).numpy()
    # possible to add rgb color channels
    color_matrix = np.zeros(shape=(output.shape[0], output.shape[0], 3))
    
    for i, row in enumerate(output):
        for j, class_idx in enumerate(row):
            color_matrix[i,j,:] = np.array(list(class_to_color[class_idx]), dtype=np.uint8)
    return color_matrix

def train(dataset, model, config):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    ce_class_weights = torch.tensor([ci.ce_weight for ci in classes], dtype=torch.float32)

    criterion = CEandDiceLoss(ce_weight=0.5,dice_weight=0.5,ce_class_weights=ce_class_weights)    
    optimizer = Adam(lr=config['lr'], params=model.parameters())
    losses = []

    for ep in tqdm(range(config['num_epochs'])):
        epoch_loss = 0
        for batch in dataloader:
            x, y = batch[0], batch[1]
            ypred = model(x)
            y = torch.tensor(y, dtype=torch.long)

            loss = criterion(ypred.to(torch.float32),y)
            losses.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if not ep % 10:
            print(f'Epoch: {ep} | Loss: {epoch_loss/len(dataloader)}')
    plt.plot(np.arange(len(losses)), losses)
    return model

def eval(dataset, model, config):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # criterion = CEandDiceLoss(ce_weight=0.5,dice_weight=0.5,ce_class_weights=ce_class_weights)    
    #losses = []
    for batch in dataloader:
        x, y = batch[0], batch[1]
        a = y.squeeze(dim=0).numpy()
        img = Image.fromarray(a).resize((224,224), Image.NEAREST)#.show()
        img.show()
        ypred = model(x)
        #loss = criterion(y,ypred)
        #losses.append(loss.detach().numpy())
        b = transform_output(ypred).astype(np.uint8)
        pred_mask = Image.fromarray(b)
        pred_mask.show()
        
if __name__ == '__main__':
    with open('hyperparams.json', 'r') as f:
        config = json.load(f)

    masks_dir = Path(os.path.join(os.getcwd(),'frames','Data','SegmentationClass'))
    img_dir = Path(os.path.join(os.getcwd(),'frames','Data', 'Images'))
    
    imgs = load_imgs(img_dir)
    masks = load_imgs(masks_dir)
    print(len(imgs), len(masks))
    
    train_data = imgs[:-3]
    train_labels = masks[:-3]
    test_data = imgs[-3:]
    test_labels = masks[-3:]

    train_dataset = ImgDataset(train_data, train_labels)
    test_dataset = ImgDataset(test_data, test_labels, trans_mask=False)

    model = SegmentationModel(num_classes=7)
    model = train(train_dataset, model, config)
    eval(test_dataset, model, config)