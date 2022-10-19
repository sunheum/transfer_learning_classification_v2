import os
import tarfile
import urllib
from glob import glob
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img

def load_dataset():
    download_root = 'https://thor.robots.ox.ac.uk/~vgg/data/pets/'
    images_url = download_root + 'images.tar.gz'
    images_path = os.path.join('datasets','images')

    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    path = os.path.join(images_path, 'images.tar.gz')
    if not os.path.isfile(path):
        urllib.request.urlretrieve(images_url, path)
        tar_file = tarfile.open(path)
        tar_file.extractall(path=images_path)
        tar_file.close()
    
    filenames = glob('./datasets/images/*.jpg')

    classes = set()
    data = []
    labels = []

    # Load the images and get the classnames from the image path
    for image in filenames:
        class_name = image.rsplit("\\", 1)[1].rsplit('_', 1)[0]
        classes.add(class_name)
        img = load_image(image)

        data.append(img)
        labels.append(class_name)

    # convert classnames to indices
    class2idx = {cl: idx for idx, cl in enumerate(classes)}        
    labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()

    data = list(zip(data, labels))
    return data, classes

class CustomDataset(Dataset):
    "Dataset to serve individual images to our model"
    
    def __init__(self, data, transforms=None):
        self.data = data
        self.len = len(data)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img, label = self.data[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
    
    def __len__(self):
        return self.len


# Since the data is not split into train and validation datasets we have to 
# make sure that when splitting between train and val that all classes are represented in both
class Databasket():
    "Helper class to ensure equal distribution of classes in both train and validation datasets"
    
    def __init__(self, data, num_cl, val_split=0.2, train_transforms=None, val_transforms=None):
        class_values = [[] for x in range(num_cl)]
        
        # create arrays for each class type
        for d in data:
            class_values[d[1].item()].append(d)
            
        self.train_data = []
        self.val_data = []
        
        # put (1-val_split) of the images of each class into the train dataset
        # and val_split of the images into the validation dataset
        for class_dp in class_values:
            split_idx = int(len(class_dp)*(1-val_split))
            self.train_data += class_dp[:split_idx]
            self.val_data += class_dp[split_idx:]
            
        self.train_ds = CustomDataset(self.train_data, transforms=train_transforms)
        self.val_ds = CustomDataset(self.val_data, transforms=val_transforms)

def get_loaders(config, input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    data, classes = load_dataset()
    databasket = Databasket(data, len(classes), val_split=0.2, 
                            train_transforms=data_transforms['train'], val_transforms=data_transforms['val'])
    
    train_loader = DataLoader(
        dataset=databasket.train_ds,
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=databasket.val_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader