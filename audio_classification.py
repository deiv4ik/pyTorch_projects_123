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
from skimage.transform import resize
import time
from torchsummary import summary
import librosa

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = pd.read_csv('quran-recitations-for-audio-classification/files_paths.csv')
data['FilePath'] = 'quran-recitations-for-audio-classification/files_paths.csv' +  data['FilePath'].str[1:]
print(data.head())

plt.figure(figsize=(8,8))
plt.pie(data['Class'].value_counts(), labels=data['Class'].value_counts().index, autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()

label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

train = data.sample(frac=0.7, random_state=7)
test = data.drop(train.index)
val = test.sample(frac=0.5, random_state=7)
test = test.drop(val.index)

print("Training Shape: ", train.shape)
print("Validation Shape: ", val.shape)
print("Testing Shape: ", test.shape)

class CustomAudioDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.labels = torch.Tensor(list(dataframe["Class"])).type(torch.LongTensor).to(device)
        self.audios = [torch.Tensor(self.get_spectrogram(path)).type(torch.FloatTensor) for path in
                       dataframe['FilePath']]

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, item):
        img_path = self.dataframe.iloc[item, 0]
        label = torch.Tensor(self.labels[item]).to(device)
        audio = self.audios[item].unsqueeze(0).to(device)
        return audio, label

    def get_spectrogram(self, file_path):
        sr = 22050
        duration = 5

        img_height = 120
        img_width = 256

        signal, sr = librosa.load(file_path, sr=sr, duration=duration)
        spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)\

        spec_resized = librosa.util.fix_length(spec_db, size=duration * sr // 512 + 1)
        spec_resized = resize(spec_resized, (img_height, img_width), anti_aliasing=True)

        return spec_resized

train_dataset = CustomAudioDataset(dataframe=train)
val_dataset = CustomAudioDataset(dataframe=val)
test_dataset = CustomAudioDataset(dataframe=test)

LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 10

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear((64*16*32), 4096)
        self.linear2 = nn.Linear(4096, 1024)
        self.linear4 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, len(data['Class'].unique()))

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.view(x.size(0), -1)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.dropout(x)

        x = self.linear4(x)
        x = self.dropout(x)

        x = self.output(x)

        return x

model = Net().to(device)
print(model)

summary(model, (1, 128, 256))
criterion = nn.CrossEntropyLoss() # Cross Entropy Loss
optimizer = Adam(model.parameters(), lr=LR) # Adam optimizer

total_loss_train_plot = [] # Empty list to be filled with train loss after each epoch
total_loss_validation_plot = [] # Empty list to be filled with validation loss after each epoch
total_acc_train_plot = [] # Empty list to be filled with train accuracy after each epoch
total_acc_validation_plot = [] # Empty list to be filled with validation accuracy after each epoch


for epoch in range(EPOCHS):
  start_time = time.time() # We use this to calculate the time of each epoch, it starts a counter once called
  total_acc_train = 0
  total_loss_train = 0
  total_loss_val = 0
  total_acc_val = 0

  for inputs, labels in train_loader:
    outputs = model(inputs)
    train_loss = criterion(outputs, labels)
    total_loss_train += train_loss.item()
    train_loss.backward()

    train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
    total_acc_train += train_acc
    optimizer.step()
    optimizer.zero_grad()

  with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        val_loss = criterion(outputs, labels)
        total_loss_val += val_loss.item()

        val_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_val += val_acc

  total_loss_train_plot.append(round(total_loss_train/1000, 4))
  total_loss_validation_plot.append(round(total_loss_val/1000, 4))
  total_acc_train_plot.append(round(total_acc_train/(train_dataset.__len__())*100, 4))
  total_acc_validation_plot.append(round(total_acc_val/(val_dataset.__len__())*100, 4))
  epoch_string = f"""
                  Epoch: {epoch+1}/{EPOCHS}, 
                  Train Loss: {round(total_loss_train/100, 4)}, 
                  Train Accuracy: {round((total_acc_train/train_dataset.__len__() * 100), 4)}, 
                  Validation Loss: {round(total_loss_val/100, 4)}, 
                  Validation Accuracy: {round((total_acc_val/val_dataset.__len__() * 100), 4)}
                  """
  print(epoch_string)
  print("="*30)

  with torch.no_grad():
      total_loss_test = 0
      total_acc_test = 0
      for indx, (input, labels) in enumerate(test_loader):
          prediction = model(input)

          acc = (torch.argmax(prediction, axis=1) == labels).sum().item()
          total_acc_test += acc

  print(f"Accuracy Score is: {round((total_acc_test / test_dataset.__len__()) * 100, 2)}%")

  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

  axs[0].plot(total_loss_train_plot, label='Training Loss')
  axs[0].plot(total_loss_validation_plot, label='Validation Loss')
  axs[0].set_title('Training and Validation Loss over Epochs')
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Loss')
  axs[1].set_ylim([0, 2])
  axs[0].legend()

  axs[1].plot(total_acc_train_plot, label='Training Accuracy')
  axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
  axs[1].set_title('Training and Validation Accuracy over Epochs')
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('Accuracy')
  axs[1].set_ylim([0, 100])
  axs[1].legend()

  plt.tight_layout()

  plt.show()






