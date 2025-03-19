import matplotlib
import opendatasets as od
import torch
from torch import nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.models.quantization import googlenet

matplotlib.use("TkAgg")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_df = pd.read_csv('bean-leaf-lesions-classification/train.csv')
val_df = pd.read_csv('bean-leaf-lesions-classification/val.csv')

data = pd.concat([train_df, val_df], ignore_index=True)
data["image:FILE"] = "bean-leaf-lesions-classification/" + data["image:FILE"]

train = data.sample(frac=0.7, random_state=7)
test = data.drop(train.index)

label_encoder = LabelEncoder()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.fit_transform(dataframe['category'])).to(device)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = (self.transform(image)/255).to(device)

        return image, label

train_dataset = CustomImageDataset(dataframe=train, transform=transform)
test_dataset = CustomImageDataset(dataframe=test, transform=transform)

n_rows = 3
n_cols = 3
f, axarr = plt.subplots(n_rows, n_cols)
for row in range(n_rows):
    for col in range(n_cols):
        image = train_dataset[np.random.randint(0,train_dataset.__len__())][0].cpu()
        axarr[row, col].imshow((image*255).squeeze().permute(1,2,0))
        axarr[row, col].axis('off')

plt.tight_layout()
plt.show()

LR = 1e-3
BATCH_SIZE = 4
EPOCHS = 15

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

googlenet_model = models.googlenet(weights='DEFAULT')
for param in googlenet_model.parameters():
    param.requires_grad = True

print(googlenet_model.fc)

num_classes = len(data["category"].unique())
googlenet_model.fc = torch.nn.Linear(googlenet_model.fc.in_features, num_classes)
print(googlenet_model.to(device))

criterion = nn.CrossEntropyLoss()
optimizer = Adam(googlenet_model.parameters(), lr=LR)

total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0

    for inputs, labels in   train_loader:
        optimizer.zero_grad()
        outputs = googlenet_model(inputs)
        train_loss = criterion(outputs, labels)
        total_loss_train += train_loss.item()
        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis=1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()

    total_loss_train_plot.append(round(total_loss_train / 1000, 4))
    total_acc_train_plot.append(round(total_acc_train / (train_dataset.__len__()) * 100, 4))
    print(
        f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {round(total_loss_train / 100, 4)} Train Accuracy {round((total_acc_train) / train_dataset.__len__() * 100, 4)}%')
    print()

with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    for indx, (input, labels) in enumerate(test_loader):

        prediction = googlenet_model(input)

        acc = (torch.argmax(prediction, axis = 1) == labels).sum().item()
        total_acc_test += acc

print(f"Accuracy Score is: {round((total_acc_test/test_dataset.__len__())*100, 2)}%")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].set_title('Training Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[1].set_ylim([0, 2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].set_title('Training Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim([0, 100])
axs[1].legend()

plt.tight_layout()

plt.show()

