import torch  
from torch import nn  
from torch.utils.data import DataLoader  
from my_dataset import MyDataset  
from train.resUnet34 import Resnet34_Unet
from torch.optim.lr_scheduler import ReduceLROnPlateau  
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print(device)

images_dir = 'img_train/images/'
masks_dir = 'img_train/masks/'
dataset = MyDataset(images_dir, masks_dir, end=2000)  
print(len(dataset))
  
val_images_dir = 'img_ver/images/'
val_masks_dir = 'img_ver/masks/'
val_dataset = MyDataset(val_images_dir, val_masks_dir , end=500)  
print(len(val_dataset))

batch_size = 32  
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
model = Resnet34_Unet().to(device)

criterion = nn.BCEWithLogitsLoss()  
lr = 0.01
min_lr = lr*0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=min_lr, verbose=True)   
num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:  
        images, labels = images.to(device), labels.to(device)   
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0  
    with torch.no_grad():  
        for images, labels in val_loader:  
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)  
            loss = criterion(outputs, labels)  
            val_loss += loss.item() * images.size(0)  
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')  
    
    if val_loss < best_val_loss:  
        best_val_loss = val_loss  
        torch.save(model.state_dict(), 'model/resunet_model.pth')  
        print("model saved")