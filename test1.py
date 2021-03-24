import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision import datasets, models, transforms
import PIL
from PIL import Image
import math
import random
import seaborn as sn
import pandas as pd
import numpy as np
from pathlib import Path
from skimage import io
import pickle
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import warnings

warnings.filterwarnings("ignore")
print("=================定义常数和容器=================")
input_size = 224
batch_size = 32
num_epoch = 1
num_workers = 4
train_acc = []
train_loss = []
val_acc = []
val_loss = []
lr_cycle = []
model_name = "resnet152"
fc_layer = 'all-st-SGD-m.9-nest-s-cycle-exp-.00001-.05-g.99994-m.8-.9'
save_best_weights_path = "./test1/save_best_weights_path"
save_last_weights_path = "./test1/save_last_weights_path"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# =================over=================
print("=================加工数据集=================")
train_dir = Path('./train/')
test_dir = Path('./test/')
class SimpsonTrainValPath():
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_val_files_path = sorted(list(self.train_dir.rglob('*.jpg')))
        self.test_path = sorted(list(self.test_dir.rglob('*.jpg')))
        self.train_val_labels = [path.parent.name for path in self.train_val_files_path]

    def get_path(self):
        train_files_path, val_files_path = train_test_split(self.train_val_files_path, test_size=0.3, \
                                                            stratify=self.train_val_labels)
        files_path = {'train': train_files_path, 'val': val_files_path}
        return files_path, self.test_path

    def get_n_classes(self):
        return len(np.unique(self.train_val_labels))


SimpsonTrainValPath = SimpsonTrainValPath(train_dir, test_dir)
train_path, test_path = SimpsonTrainValPath.get_path()


class SimpsonsDataset(Dataset):

    def __init__(self, files_path, data_transforms):
        self.files_path = files_path
        self.transform = data_transforms

        if 'test' not in str(self.files_path[0]):
            self.labels = [path.parent.name for path in self.files_path]
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):

        img_path = str(self.files_path[idx])
        image = Image.open(img_path)
        image = self.transform(image)

        if 'test' in str(self.files_path[0]):
            return image
        else:
            label_str = str(self.files_path[idx].parent.name)
            label = self.label_encoder.transform([label_str]).item()

            return image, label


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(contrast=0.9),
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomApply([transforms.RandomHorizontalFlip(p=1), transforms.ColorJitter(contrast=0.9)], p=0.5),
            transforms.RandomApply([transforms.RandomHorizontalFlip(p=1), transforms.ColorJitter(brightness=0.1)],
                                   p=0.5),
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {mode: SimpsonsDataset(train_path[mode], data_transforms[mode]) for mode in ['train', 'val']}
dataloaders_dict = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)}

image_datasets_test = SimpsonsDataset(test_path, data_transforms['val'])
dataloader_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# =================over=================

print("=================定义模型=================")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    if model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft


model_ft = initialize_model(model_name, SimpsonTrainValPath.get_n_classes(), False, use_pretrained=True)
model_ft = model_ft.to(device)
# =================over=================
print("=================定义损失函数=================")
criterion = nn.CrossEntropyLoss()
# =================over=================
print("=================定义优化器=================")
base_lr = 0.0012
max_lr = 0.0022
lr_find_epochs = 1
step_size = lr_find_epochs * len(dataloaders_dict['train'])
params_to_update = model_ft.parameters()
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, nesterov=True)
scheduler = optim.lr_scheduler.CyclicLR(optimizer_ft, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size,
                                        mode='exp_range', gamma=0.99994, scale_mode='cycle', cycle_momentum=True,
                                        base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
# =================over=================
print("=================运行=================")
def train_model(model, dataloaders, criterion, optimizer, save_best_weights_path, save_last_weights_path,
                num_epochs=1, is_inception=False):
    since = time.time()
    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    lr_find_lr = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm_notebook(dataloaders[phase]):
                input = input.to(device)
                label = input.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        lr_step = optimizer_ft.state_dict()["param_groups"][0]["lr"]
                        lr_find_lr.append(lr_step)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    history_val = {'loss': val_loss_history, 'acc': val_acc_history}
    history_train = {'loss': train_loss_history, 'acc': train_acc_history}
    return model, history_val, history_train, time_elapsed, lr_find_lr, best_acc
for i in range(num_epoch):
    model, history_val, history_train, time_elapsed, lr_find_lr, best_acc = train_model(model_ft, dataloaders_dict,
                                                                                        criterion, optimizer_ft,
                                                                                        save_best_weights_path,
                                                                                        save_last_weights_path,
                                                                                        num_epochs=1,
                                                                                        is_inception=(
                                                                                                    model_name == "inception"))
    val_loss += history_val['loss']
    val_acc += history_val['acc']
    train_loss += history_train['loss']
    train_acc += history_train['acc']
    lr_cycle += lr_find_lr
# =================over=================
print("=================可视化=================")
def imshow(inp, title=None, plt_ax=plt, default=False):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)
def visualization(train, val, is_loss=True):
    if is_loss:
        plt.figure(figsize=(17, 10))
        plt.plot(train, label='Training loss')
        plt.plot(val, label='Val loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    else:
        plt.figure(figsize=(17, 10))
        plt.plot(train, label='Training acc')
        plt.plot(val, label='Val acc')
        plt.title('Training and validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.show()
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8),sharey=True, sharex=True)
for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0, 4500))
    im_val, label = image_datasets['train'][random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                image_datasets['val'].label_encoder.inverse_transform([label])[0].split('_')))
    imshow(im_val.data.cpu(), title=img_label,plt_ax=fig_x)
plt.plot(lr_cycle)
visualization(train_acc, val_acc, is_loss = False)
visualization(train_loss, val_loss, is_loss = True)
def confusion_matrix():
    actual = [image_datasets['val'][i][1] for i in range(len(image_datasets['val']))]
    image = [image_datasets['val'][i][0] for i in range(len(image_datasets['val']))]
    img_conf_dataloader = torch.utils.data.DataLoader(image, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs = predict(model_ft, img_conf_dataloader)
    preds = np.argmax(probs, axis=1)
    df = pd.DataFrame({'actual': actual, 'preds': preds})
    confusion_matrix = pd.crosstab(df['actual'], df['preds'], rownames=['Actual'], colnames=['Predicted'],margins=False)
    label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
    yticklabels = label_encoder.classes_
    plt.subplots(figsize=(20, 20))
    sn.heatmap(confusion_matrix, annot=True, fmt="d", linewidths=0.5, cmap="YlGnBu", cbar=False, vmax=30,
               yticklabels=yticklabels, xticklabels=yticklabels);
confusion_matrix()
# =================over=================
print("=================预测并提交=================")
def predict(model, test_loader):
    with torch.no_grad():
        logits = []
        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
    probs = nn.functional.softmax(torch.cat(logits), dim=1).numpy()
    return probs
probs = predict(model_ft, dataloader_test)
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
preds = label_encoder.inverse_transform(np.argmax(probs, axis = 1 ))
test_filenames = [path.name for path in image_datasets_test.files_path]
my_submit = pd.DataFrame({'Id': test_filenames, 'Expected': preds})
my_submit.head()
# =================over=================