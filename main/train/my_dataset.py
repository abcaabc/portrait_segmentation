import os  
from torch.utils.data import Dataset  
import cv2
from torchvision import transforms  
import numpy as np

class MyDataset(Dataset): 
     
    def __init__(self, images_dir, masks_dir, start=0,end=None):   

        self.images_dir = images_dir    
        self.masks_dir = masks_dir    
        self.image_filenames = os.listdir(images_dir)[start:end]
        self.mask_filenames = os.listdir(masks_dir)[start:end]
    
        self.img_transform = transforms.Compose([  
            transforms.ToTensor(),  
            transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) 
        self.mask_transform = transforms.Compose([    
            transforms.ToTensor(),  
            transforms.Lambda(lambda tensor: (tensor > 0).float()), 
        ])  
    def __len__(self):  
        return len(self.image_filenames)  

    def resize_image(self, image, size=(256, 256)):  
        imsize = image.shape  
        ratio = min(size[0] / imsize[1], size[1] / imsize[0])  
        new_width = int(imsize[1] * ratio)  
        new_height = int(imsize[0] * ratio)  
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  
        
        left = (size[0] - new_width) // 2  
        top = (size[1] - new_height) // 2  
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            new_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)  
        elif len(image.shape) == 2:
            new_image = np.zeros((size[1], size[0]), dtype=np.uint8)  
        else:  
            raise ValueError(f"不支持的图片尺寸: {image.shape}")  
        
        new_image[top:top+new_height, left:left+new_width] = resized_image  
        return new_image 

    def __getitem__(self, idx):  
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])  
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])  
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2RGB)

        image = self.resize_image(image)  
        mask = self.resize_image(mask)  

        image = self.img_transform(image)  
        mask = self.mask_transform(mask)
        return image, mask
