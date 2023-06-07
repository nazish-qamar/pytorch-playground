import torch
import torchvision
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader


class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y = torch.from_numpy(xy[:, [0]])  # size will become [n_samples, 1]
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # this will allow indexing
        # dataset
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
#features, labels = dataset[0]
#print(features, labels)

data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True#,
                         #num_workers=2
                         )

data = next(iter(data_loader))
features, labels = data
print(features, labels)

# for training we will loop like:
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(data_loader):
        #forward, backward pass, update
        if (i+1) % 5 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step{i+1}/{n_iterations}, inputs {inputs.shape}")

#we can also download datasets from torchvision
#torchvision.datasets.MNIST() or fashion-mnist, cifar, coco