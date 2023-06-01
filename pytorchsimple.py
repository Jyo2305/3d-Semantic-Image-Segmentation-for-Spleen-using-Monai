import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import sys
import pytorchplottingsimple as ptplot
import torch.nn.functional as F
from torch import concatenate

class model_builder(nn.Module):
    def __init__(self, input_dim, dnn):
        super().__init__()
        self.model = nn.Sequential()
        self.count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        tmp = dnn[0].split(" ")
        
        if tmp[0]=='lin':
                self.count[0] += 1
                name = 'linear' + '_' + str(self.count[0])
                input_dim = int(input_dim)
                output = int(tmp[1])
                self.model.add_module(name, nn.Linear(in_features=input_dim, out_features=output))
        elif tmp[0]=='conv':
                self.count[3] += 1
                name = 'conv' + '_' + str(self.count[3])
                if len(tmp) > 2:
                    ksz = int(tmp[2])
                    try:
                        st = int(tmp[3])
                    except:
                        st = 0
                    try:
                        if tmp[4] == 'same':
                            pad='same'
                        else:
                            pad = int(tmp[4])
                    except:
                        pad = 0                    
                else:
                    ksz = 3
                    st=0
                    pad=0
                output = int(tmp[1])
                self.model.add_module(name, nn.Conv2d(in_channels=input_dim, out_channels=output, 
                                                      kernel_size=ksz, stride=st, padding=pad, bias=False))
        elif tmp[0]=='conv3':
                self.count[3] += 1
                name = 'conv3' + '_' + str(self.count[3])
                if len(tmp) > 2:
                    ksz = int(tmp[2])
                    try:
                        st = int(tmp[3])
                    except:
                        st = 0
                    try:
                        if tmp[4] == 'same':
                            pad='same'
                        else:
                            pad = int(tmp[4])
                    except:
                        pad = 0                    
                else:
                    ksz = 3
                    st=0
                    pad=0
                output = int(tmp[1])
                self.model.add_module(name, nn.Conv3d(in_channels=input_dim, out_channels=output, 
                                                      kernel_size=ksz, stride=st, padding=pad, bias=False))
        
        elif tmp[0]=='flat':
                self.count[5] += 1
                name = 'flatten' + '_' + str(self.count[5])
                self.model.add_module(name, nn.Flatten())
                output = int(tmp[1]) * int(tmp[1])
        elif tmp[0]=='convtrans':
                self.count[12] += 1
                name = 'convtranspose2d' + '_' + str(self.count[12])
                if len(tmp) > 2:
                    ksz = int(tmp[2])
                    try:
                        st = int(tmp[3])
                    except:
                        st = 0
                    try:
                        if tmp[4] == 'same':
                            pad='same'
                        else:
                            pad = int(tmp[4])
                    except:
                        pad = 0                    
                else:
                    ksz = 3
                    st=0
                    pad=0
                output = int(tmp[1])
                self.model.add_module(name, nn.ConvTranspose2d(in_channels=input_dim, out_channels=output, 
                                                      kernel_size=ksz, stride=st, padding=pad, bias=False))
        elif tmp[0]=='convtrans3':
                self.count[12] += 1
                name = 'convtranspose2d' + '_' + str(self.count[12])
                if len(tmp) > 2:
                    ksz = int(tmp[2])
                    try:
                        st = int(tmp[3])
                    except:
                        st = 0
                    try:
                        if tmp[4] == 'same':
                            pad='same'
                        else:
                            pad = int(tmp[4])
                    except:
                        pad = 0                    
                else:
                    ksz = 3
                    st=0
                    pad=0
                output = int(tmp[1])
                self.model.add_module(name, nn.ConvTranspose3d(in_channels=input_dim, out_channels=output, 
                                                      kernel_size=ksz, stride=st, padding=pad, bias=False))
        
        for i in dnn[1:]:
            tmp = i.split(" ")
            if tmp[0]=='lin':
                self.count[0] += 1
                name = 'linear' + '_' + str(self.count[0])
                outputnew = int(tmp[1])
                self.model.add_module(name, nn.Linear(in_features=output, out_features=outputnew))
                output = int(tmp[1])
            elif tmp[0]=='rel':
                self.count[1] += 1
                name = 'relu' + '_' + str(self.count[1])
                self.model.add_module(name, nn.ReLU(True))
            elif tmp[0]=='drop':
                self.count[2] += 1
                name = 'dropout' + '_' + str(self.count[2])
                dvalue = float(tmp[1])
                self.model.add_module(name, nn.Dropout(dvalue))
            elif tmp[0]=='conv':
                self.count[3] += 1
                name = 'conv' + '_' + str(self.count[3])
                if len(tmp) > 2:
                    ksz = int(tmp[2])
                    try:
                        st = int(tmp[3])
                    except:
                        st = 0
                    try:
                        if tmp[4] == 'same':
                            pad='same'
                        else:
                            pad = int(tmp[4])
                    except:
                        pad = 0                    
                else:
                    ksz = 3
                    st=0
                    pad=0
                outputnew = int(tmp[1])
                self.model.add_module(name, nn.Conv2d(in_channels=output, out_channels=outputnew,
                                                      kernel_size=ksz, stride=st, padding=pad, bias=False))
                output = int(tmp[1])
            elif tmp[0]=='conv3':
                self.count[13] += 1
                name = 'conv3' + '_' + str(self.count[13])
                if len(tmp) > 2:
                    ksz = int(tmp[2])
                    try:
                        st = int(tmp[3])
                    except:
                        st = 0
                    try:
                        if tmp[4] == 'same':
                            pad='same'
                        else:
                            pad = int(tmp[4])
                    except:
                        pad = 0                    
                else:
                    ksz = 3
                    st=0
                    pad=0
                outputnew = int(tmp[1])
                self.model.add_module(name, nn.Conv3d(in_channels=output, out_channels=outputnew, 
                                                      kernel_size=ksz, stride=st, padding=pad, bias=False))
            elif tmp[0]=='convtrans':
                self.count[13] += 1
                name = 'convtrans' + '_' + str(self.count[13])
                if len(tmp) > 2:
                    ksz = int(tmp[2])
                    try:
                        st = int(tmp[3])
                    except:
                        st = 0
                    try:
                        if tmp[4] == 'same':
                            pad='same'
                        else:
                            pad = int(tmp[4])
                    except:
                        pad = 0                    
                else:
                    ksz = 3
                    st=0
                    pad=0
                outputnew = int(tmp[1])
                self.model.add_module(name, nn.ConvTranspose2d(in_channels=output, out_channels=outputnew, 
                                                      kernel_size=ksz, stride=st, padding=pad, bias=False))
            elif tmp[0]=='convtrans3':
                self.count[12] += 1
                name = 'convtranspose3d' + '_' + str(self.count[12])
                if len(tmp) > 2:
                    ksz = int(tmp[2])
                    try:
                        st = int(tmp[3])
                    except:
                        st = 0
                    try:
                        if tmp[4] == 'same':
                            pad='same'
                        else:
                            pad = int(tmp[4])
                    except:
                        pad = 0                    
                else:
                    ksz = 3
                    st=0
                    pad=0
                outputnew = int(tmp[1])
                self.model.add_module(name, nn.ConvTranspose3d(in_channels=output, out_channels=outputnew, 
                                                      kernel_size=ksz, stride=st, padding=pad, bias=False))
                output = outputnew
            elif tmp[0]=='mpl2':
                self.count[4] += 1
                name = 'maxpool2d' + '_' + str(self.count[4])
                self.model.add_module(name, nn.MaxPool2d(kernel_size=2))
            elif tmp[0]=='mpl3':
                self.count[4] += 1
                name = 'maxpool3d' + '_' + str(self.count[4])
                self.model.add_module(name, nn.MaxPool3d(kernel_size=2))
            elif tmp[0]=='flat':
                self.count[5] += 1
                name = 'flatten' + '_' + str(self.count[5])
                self.model.add_module(name, nn.Flatten())
                output = int(tmp[1])
            elif tmp[0]=='batch2':
                self.count[6] += 1
                if len(tmp)>1:
                    value = int(tmp[1])
                else:
                    value = output
                name = 'batch2d' + '_' + str(self.count[6])
                self.model.add_module(name, nn.BatchNorm2d(value))
            elif tmp[0]=='batch3':
                self.count[6] += 1
                if len(tmp)>1:
                    value = int(tmp[1])
                else:
                    value = output
                name = 'batch3d' + '_' + str(self.count[6])
                self.model.add_module(name, nn.BatchNorm3d(value))
            elif tmp[0]=='batch1':
                self.count[7] += 1
                name = 'batch1d' + '_' + str(self.count[7])
                self.model.add_module(name, nn.BatchNorm1d(num_features=output))
            elif tmp[0]=='inst2':
                self.count[8] += 1
                name = 'instance2d' + '_' + str(self.count[8])
                self.model.add_module(name, nn.InstanceNorm2d(num_features=output))
            elif tmp[0]=='inst3':
                self.count[8] += 1
                name = 'instance3d' + '_' + str(self.count[8])
                self.model.add_module(name, nn.InstanceNorm3d(num_features=output))
            elif tmp[0]=='inst1':
                self.count[9] += 1
                name = 'instance1d' + '_' + str(self.count[9])
                self.model.add_module(name, nn.InstanceNorm1d(num_features=output))
            elif tmp[0]=='lkrel':
                self.count[10] += 1
                name = 'Leakyrelu' + '_' + str(self.count[10])
                self.model.add_module(name, nn.LeakyReLU(0.2, inplace=True))
            elif tmp[0]=='sigm':
                self.count[11] += 1
                name = 'sigmoid' + '_' + str(self.count[11])
                self.model.add_module(name, nn.Sigmoid())
            elif tmp[0]=='tan':
                self.count[14] += 1
                name = 'Tanh' + '_' + str(self.count[14])
                self.model.add_module(name, nn.Tanh())

    def forward(self, x):
        return self.model(x)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

def train_step(model: torch.nn.Module, data_loader: DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    train_loss, train_acc = 0, 0
    for batch, (x, y) in enumerate(data_loader):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc
    
def test_step(model: torch.nn.Module, data_loader: DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    test_loss, test_acc = 0, 0
    best_acc = 0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            test_pred = model(x)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_pred_class = torch.argmax(test_pred, dim=1)
            test_acc += (test_pred_class == y).sum().item()/len(test_pred)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc

def binary_class_train(epochs, model, loss_fn, optimizer, x_train, x_test, y_train, y_test, path = os.getcwd(), name = 'model'):
    results = {'epoch_values': [],
               'train_loss': [],
               'test_loss': [],
               'train_acc': [],
               'test_acc': []}
    best_acc = 0
    name1 = name + ".pt"
    name2 = name + '_dict.pt'
    path1 = os.path.join(path, name1)
    path2 = os.path.join(path, name2)
    for epoch in range(1, epochs+1, 1):
        # Main Training
        model.train()
        y_logits = model(x_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # for multiclass, use softmax # y_pred=torch.softmax(y_logits,dim=1).argmax(dim=1)
        loss_train = loss_fn(y_logits, y_train) # If loss function is BCELoss, then use torch.sigmoid(y_logits) instead
                                                # If wrong datatype, convert y_train, y_test to torch.longTensor
        loss_train_tmp = loss_train
        acc = accuracy_fn(y_train, y_pred)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        # Main Testing
        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits)) # For multiclass, use softmax
            loss_test = loss_fn(test_logits,y_test) # If loss function is BCELoss, then use torch.sigmoid(test_logits) instead
            loss_test = round(float(loss_test), 8)
            test_acc = accuracy_fn(y_test, test_pred)
            loss_train_tmp = round(float(loss_train_tmp), 8)
            print(f"Epoch: {epoch} | Train Loss: {loss_train_tmp} | Train Acc: {acc} | Test Loss: {loss_test} | Test Acc: {test_acc}")
        
        results['epoch_values'].append(epoch)
        loss_train = loss_train.detach().numpy()
        results['train_loss'].append(loss_train)
        results['test_loss'].append(loss_test)
        results['train_acc'].append(acc)
        results['test_acc'].append(test_acc)
        if test_acc > best_acc:
            torch.save(model, path1)
            torch.save(model.state_dict(), path2)
            best_acc = test_acc
    return results

def multi_class_training(epochs,
                        model: torch.nn.Module, train_dataloader: DataLoader,
                        test_dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                        loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
                        path = os.getcwd(),
                        name = 'model'
                        ):
    results = {'epoch_values': [],
               'train_loss': [],
               'test_loss': [],
               'train_acc': [],
               'test_acc': []}
    best_acc = 0
    name1 = name + ".pt"
    name2 = name + '_dict.pt'
    path1 = os.path.join(path, name1)
    path2 = os.path.join(path, name2)
    for epoch in tqdm(range(1, epochs+1)):
        train_loss, train_acc = train_step(model = model, data_loader=train_dataloader,
                                           loss_fn=loss_fn, optimizer=optimizer)
        test_loss, test_acc = test_step(model = model, data_loader=test_dataloader,
                                           loss_fn=loss_fn, optimizer=optimizer)
        print(f"Epoch: {epoch} | Train Loss: {train_loss: .5f} | Train Acc: {train_acc: .5f} | Test Loss: {test_loss: .5f} | Test Acc: {test_acc: .5f}")
        results['epoch_values'].append(epoch)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        if test_acc > best_acc:
            torch.save(model, path1)
            torch.save(model.state_dict(), path2)
            best_acc = test_acc
    print(f"Saved at {path1}")
    print(f"Saved at {path2}")
    return results
    
def predict(model, test_set, classes, type='m'):
    
    test_pred = model(test_set)
    pred_values = []
    pred_labels = []
    if type=='b':
        test_pred = torch.round(torch.sigmoid(test_pred))
        test_pred = test_pred.detach().numpy()
        for t in test_pred:
            if t >= 0.5:
                pred_values.append(1)
            else:
                pred_values.append(0)
    elif type=='m':
        test_pred=torch.softmax(test_pred,dim=1).argmax(dim=1)
        test_pred = np.array(test_pred)
        for t in test_pred:
            pred_values.append(t)
    for pred in pred_values:
        pred_labels.append(classes[pred])
    return pred_values, pred_labels

def create_transform(img_size, hp, stats=0):
    img_size = img_size
    elements = []

    elements.append(transforms.Resize(size=(img_size, img_size)))
    for i in hp:
        tmp = i.split(" ")
        if tmp[0] == "hflip":
            tmp[1] = float(tmp[1])
            elements.append(transforms.RandomHorizontalFlip(p = tmp[1]))
        elif tmp[0] == "taug":
            tmp[1] = int(tmp[1])
            elements.append(transforms.TrivialAugmentWide(num_magnitude_bins=tmp[1]))
        elif tmp[0] == 'tens':
            elements.append(transforms.ToTensor())
        elif tmp[0] == "vflip":
            tmp[1] = float(tmp[1])
            elements.append(transforms.RandomVerticalFlip(p = tmp[1]))
        elif tmp[0] == 'rot':
            tmp[1] = float(tmp[1])
            elements.append(transforms.RandomRotation(degrees=tmp[1]))
        elif tmp[0] == 'cencrop':
            elements.append(transforms.CenterCrop(img_size))
        elif tmp[0] == 'norm':
            mean, std = stats
            elements.append(transforms.Normalize(mean=mean, std=std))
            
    data_transform = transforms.Compose(elements)
    return data_transform

def create_dataloader(train_dir, test_dir, data_transform, batch):
    train_data = datasets.ImageFolder(root = train_dir,
                                    transform = data_transform,
                                    target_transform=None
                                    )
    
    test_data = datasets.ImageFolder(root = test_dir,
                                    transform = data_transform
                                    )
    
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch,
                                num_workers=0,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch,
                                num_workers=0,
                                shuffle=False)
    class_names = train_data.classes
    return train_dataloader, test_dataloader, class_names

def image_classification(idim, model_layers, args, type='m'):
    ### Multi image classification
    batch_size, train_dir, test_dir, img_size, transform_parameters, optim, result_path, model_name, epochs = args
    if test_dir == 'same':
        test_dir = train_dir
    
    path = result_path
    model = model_builder(idim, model_layers)
    dtransform = create_transform(img_size, transform_parameters)

    train_data, test_data, classes = create_dataloader(train_dir, test_dir, dtransform, batch_size)

    if type=='m' or type=='multi' or type=='categorical':
        loss_fn = nn.CrossEntropyLoss()
    elif type=='b' or type=='binary':
        loss_fn = nn.BCEWithLogitsLoss()
    
    optim = optim.split(" ")
    if optim[0] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = float(optim[1]))
    elif optim[0] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = float(optim[1]))
    elif optim[0] == 'RMS':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = float(optim[1]))

    history = multi_class_training(epochs, model, train_data, test_data, optimizer, loss_fn, path, model_name)

    ptplot.training_plot(history, result_path, 'loss')
    ptplot.training_plot(history, result_path, 'acc')
    return history

def check_confusion_matrix(model, img_size, img_dir, img_count, result_path):
    real_y = []
    dtransform = create_transform(img_size, ['tens'])
    train_data, test_data, classes = create_dataloader(img_dir, img_dir, dtransform, img_count)

    x, y = next(iter(train_data))
    pred, pred_classes = predict(model, x, classes, 'm')
    for j in y:
         real_y.append(classes[j])
    ptplot.cm_plot(result_path, real_y, pred_classes)
    
def img_denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]

def train_discriminator(real_images, discriminator, generator, opt_d, batch_size, latent_size, device):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(generator, discriminator, opt_g, batch_size, latent_size, device):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

def gan_model_training(train_dataloader, generator, discriminator, opt_d, opt_g, fixed_latent,
                       batch_size, latent_size, epochs, save_dir, stats, device, start_idx=1):
    results = {
    'losses_g': [],
    'losses_d': [],
    'real_scores': [],
    'fake_scores': []
    }
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dataloader):
            loss_d, real_score, fake_score = train_discriminator(real_images, discriminator, generator,
                                                                 opt_d, batch_size, latent_size, device)
            loss_g = train_generator(generator, discriminator, opt_g, batch_size, latent_size, device)
            
            # Recording losses
        results['losses_g'].append(loss_g)
        results['losses_d'].append(loss_d)
        results['real_scores'].append(real_score)
        results['fake_scores'].append(fake_score)
            
        print(f"Epoch: {epoch+1}/{epochs} | loss_g: {loss_g: .5f} | loss_d: {loss_d: .5f} | real_score: {real_score: .4f} | fake_score: {fake_score: .4f}")
            
        ptplot.save_samples(generator, save_dir, epoch+start_idx, fixed_latent, stats, show=False)
        save_epoch = str(epoch+1)
    torch.save(generator, os.path.join(save_dir, 'gan_generator_' + save_epoch + '.pt'))
    torch.save(discriminator, os.path.join(save_dir, 'gan_dicriminator_' + save_epoch + '.pt'))
    torch.save(generator.state_dict(), os.path.join(save_dir, 'gan_generator_dict_' + save_epoch + '.pt'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'gan_dicriminator_dict_' + save_epoch + '.pt'))
    return results

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


    
        
    
    
    












