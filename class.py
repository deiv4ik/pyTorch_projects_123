import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = pd.read_csv('rice-type-classification/riceClassification.csv')

data.dropna(inplace=True)
data.drop(['id'], axis=1, inplace=True)
print(data.head())

original = data.copy()
for column in data.columns:
    data[column] = data[column] / data[column].abs().max()

X = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.5)

class dataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

training_data = dataset(X_train, y_train)
validation_data = dataset(X_val, y_val)
testing_data = dataset(X_test, y_test)

train_dataloader = DataLoader(training_data, batch_size=8, shuffle= True)
validation_dataloader = DataLoader(validation_data, batch_size=8, shuffle= True)
testing_dataloader = DataLoader(testing_data, batch_size=8, shuffle= True)

HIDDEN_NEURONS = 10
class MyModel(nn.Module):
    def __init__(self):

        super(MyModel, self).__init__()

        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.linear = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

model = MyModel().to(device)
summary(model, (X.shape[1],))

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

epochs = 10
for epoch in range(epochs):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    for data in train_dataloader:
        inputs, labels = data

        prediction = model(inputs).squeeze(1)

        batch_loss = criterion(prediction, labels)

        total_loss_train += batch_loss.item()

        acc = ((prediction).round() == labels).sum().item()

        total_acc_train += acc

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data

            prediction = model(inputs).squeeze(1)

            batch_loss = criterion(prediction, labels)

            total_loss_val += batch_loss.item()

            acc = ((prediction).round() == labels).sum().item()

            total_acc_val += acc

    total_loss_train_plot.append(round(total_loss_train / 1000, 4))
    total_loss_validation_plot.append(round(total_loss_val / 1000, 4))
    total_acc_train_plot.append(round(total_acc_train / (training_data.__len__()) * 100, 4))
    total_acc_validation_plot.append(round(total_acc_val / (validation_data.__len__()) * 100, 4))

    print(f'''Epoch no. {epoch + 1} Train Loss: {total_loss_train / 1000:.4f} Train Accuracy: {(total_acc_train / (training_data.__len__()) * 100):.4f} Validation Loss: {total_loss_val / 1000:.4f} Validation Accuracy: {(total_acc_val / (validation_data.__len__()) * 100):.4f}''')
    print("=" * 50)

with torch.no_grad():
  total_loss_test = 0
  total_acc_test = 0
  for data in testing_dataloader:
    inputs, labels = data

    prediction = model(inputs).squeeze(1)

    batch_loss_test = criterion((prediction), labels)
    total_loss_test += batch_loss_test.item()
    acc = ((prediction).round() == labels).sum().item()
    total_acc_test += acc

print(f"Accuracy Score is: {round((total_acc_test/X_test.shape[0])*100, 2)}%")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_validation_plot, label='Validation Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0, 2])
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

area = float(input("Area: "))/original['Area'].abs().max()
MajorAxisLength = float(input("Major Axis Length: "))/original['MajorAxisLength'].abs().max()
MinorAxisLength = float(input("Minor Axis Length: "))/original['MinorAxisLength'].abs().max()
Eccentricity = float(input("Eccentricity: "))/original['Eccentricity'].abs().max()
ConvexArea = float(input("Convex Area: "))/original['ConvexArea'].abs().max()
EquivDiameter = float(input("EquivDiameter: "))/original['EquivDiameter'].abs().max()
Extent = float(input("Extent: "))/original['Extent'].abs().max()
Perimeter = float(input("Perimeter: "))/original['Perimeter'].abs().max()
Roundness = float(input("Roundness: "))/original['Roundness'].abs().max()
AspectRation = float(input("AspectRation: "))/original['AspectRation'].abs().max()

my_inputs = [area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation]

print("="*20)
model_inputs = torch.Tensor(my_inputs).to(device)
prediction = (model(model_inputs))
print(prediction)
print("Class is: ", round(prediction.item()))
