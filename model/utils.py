from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import transforms
@dataclass
class ClassItem:
    index: int
    name: str
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

class ImgDataset(Dataset):
    def __init__(self, imgs : list, masks : list):
        super().__init__()
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
        return len(self.imgs)

    def __getitem__(self, index):
        return self.preprocess(self.imgs[index]), self.masks[index]
    
    