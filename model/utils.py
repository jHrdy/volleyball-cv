from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
from PIL import Image

@dataclass
class ClassItem:
    name: str
    index: int
    ce_weight: int
    rgb: tuple[int,int,int]

@dataclass
class Classes:
    def __init__(self, class_items: list[ClassItem]):
        self.class_items = class_items
        for item in class_items:
            setattr(self, item.name, item)
    
    def __iter__(self):
        return iter(self.class_items)

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

def transform_mask(mask):
    mask = mask.resize((224,224), Image.NEAREST)
    mask = np.array(mask)
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor()
    # ])
    # mask = preprocess(mask) * 255
    # mask = mask.type(torch.int32).permute(1,2,0).numpy()
    class_matrix = np.zeros((mask.shape[0], mask.shape[0]))

    colors = set()
    for i, row in enumerate(mask):
        for j, pixel in enumerate(row):
            colors.add(tuple(pixel))
            class_matrix[i,j] = color_to_class[tuple(pixel)]
    # print(colors)
    # print(color_to_class)
    # exit()
    return class_matrix.astype(np.int32)

class ImgDataset(Dataset):
    def __init__(self, imgs : list, masks : list, trans_mask=True):
        super().__init__()
        self.trans_mask = trans_mask
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.imgs = imgs
        self.masks = masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        img = self.imgs[index].resize((224,224), Image.NEAREST)
        img = torch.tensor(np.array(img, dtype=np.float32), dtype=torch.float32).permute(2,0,1)
        return img, transform_mask(self.masks[index]) if self.trans_mask else np.array(self.masks[index])
    
    