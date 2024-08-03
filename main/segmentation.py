import torch  
import cv2  
import numpy as np  
from torchvision import transforms  
import os
from train.resUnet34 import Resnet34_Unet

def resize_image(image, size=(256, 256)):  
    h, w, _ = image.shape  
    ratio = min(size[0] / w, size[1] / h)  
    new_width = int(w * ratio)  
    new_height = int(h * ratio)  
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  

    left = (size[0] - new_width) // 2  
    top = (size[1] - new_height) // 2  

    if len(image.shape) == 3 and image.shape[2] == 3: 
        new_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)  
    elif len(image.shape) == 2:
        new_image = np.zeros((size[1], size[0]), dtype=np.uint8)  
    else:  
        raise ValueError(f"Unsupported image shape: {image.shape}")  

    new_image[top:top + new_height, left:left + new_width] = resized_image  
    return new_image  

def preprocess_image(image):  
    image_resize = resize_image(image)  
    img_transform = transforms.Compose([  
        transforms.ToTensor(),  
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        lambda image: image.unsqueeze(0)
    ]) 
    image_tensor = img_transform(image_resize)
    return image_tensor,image_resize 

def segment_image(image_tensor, device, threshold=0.8):  
    with torch.no_grad():  
        model = Resnet34_Unet().to(device)  
        # print(os.getcwd())
        model_path = r'\train\model\resunet_model.pth'
        path = os.getcwd()+model_path
        model.load_state_dict(torch.load(path, map_location=device))  
        model.eval()  
        logits = model(image_tensor)  
        output_pred = (logits > threshold).squeeze().cpu().numpy()  
    return output_pred  

def remove_bg(fg_image, output_pred):  
    alpha = np.zeros_like(output_pred, dtype=np.float32)  
    alpha[output_pred] = 1.0
    alpha_expanded = np.expand_dims(alpha, axis=2)  
    output_img = alpha_expanded * fg_image.astype(np.float32)  
    return output_img.astype(np.uint8)

def segment(img_path):
    fg_image = cv2.imread(img_path)
    image_tensor,image_resize = preprocess_image(fg_image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)  
    output_pred = segment_image(image_tensor,device)  
    output_image = remove_bg(image_resize, output_pred)  
    success, buffer = cv2.imencode('.PNG', output_image)
    return success, buffer