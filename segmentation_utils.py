
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged, Resized, ToTensord, Spacingd, 
                              Orientationd, ScaleIntensityRanged, CropForegroundd)
import torch
from monai.losses import DiceLoss
import pytorchplottingsimple as ptplot

def create_monai_transformer(type, keys, args):
    pixdim, spatial_size, a_min, a_max = args
    transformer_params = []
    for i in type:
        if i == 'ensurech':
            transformer_params.append(EnsureChannelFirstd(keys=keys))
        elif i == 'scaleint':
            transformer_params.append(ScaleIntensityRanged(keys=[keys[0]], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True))
        elif i == 'load':
            transformer_params.append(LoadImaged(keys=keys))
        elif i == 'cropfore':
            transformer_params.append(CropForegroundd(keys=keys, source_key=keys[0]))
        elif i == 'space':
            if len(keys)==2:
                transformer_params.append(Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")))
            else:
                transformer_params.append(Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear")))
        elif i == 'resize':
            transformer_params.append(Resized(keys=keys, spatial_size=spatial_size))
        elif i == 'orient':
            transformer_params.append(Orientationd(keys=keys, axcodes="RAS"))
        elif i == 'tens':
            transformer_params.append(ToTensord(keys=keys))
    
    return transformer_params
    

def create_dataloader(input_directory, pixdim, a_min, a_max, spatial_size, train_test_folders, transform_type, cache=False):
    """
    Data Preparation by monai data transformation and convertion into torch dataloaders
    """
    
    set_determinism(seed=0)
    keys = ["volume", "segment"]
    
    ## Getting train and test data paths
    train_volumes_path = sorted(glob(os.path.join(input_directory, train_test_folders[0], "*.nii.gz")))
    train_segmentation_path = sorted(glob(os.path.join(input_directory, train_test_folders[1], "*.nii.gz")))

    test_volumes_path = sorted(glob(os.path.join(input_directory, train_test_folders[2], "*.nii.gz")))
    if len(train_test_folders)>3:
        test_segmentation_path = sorted(glob(os.path.join(input_directory, train_test_folders[3], "*.nii.gz")))

    ## separating volume paths into image_name and segmentation paths into label_name and then putting them into dictionary
    train_files = [{"volume": image_name, "segment": label_name} 
                   for image_name, label_name in zip(train_volumes_path, train_segmentation_path)]
    if len(train_test_folders)>3:
        test_files = [{"volume": image_name, "segment": label_name} 
                        for image_name, label_name in zip(test_volumes_path, test_segmentation_path)]
    else:
        test_files = [{"volume": image_name} for image_name in test_volumes_path]

    ## Creating the transform to be applied on the train and test data
    args = pixdim, spatial_size, a_min, a_max
    params1 = create_monai_transformer(type=transform_type, keys=keys, args=args)
    if len(train_test_folders)>3:
        params2 = params1
    else:
        keys2 = ["volume"]
        params2 = create_monai_transformer(type=transform_type, keys=keys2, args=args)
    train_transformer = Compose(params1)
    
    test_transformer = Compose(params2)
    
    ## Creating the dataloader
    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transformer,cache_rate=1.0)
        train_dl = DataLoader(train_ds, batch_size=1)
        test_ds = CacheDataset(data=test_files, transform=test_transformer, cache_rate=1.0)
        test_dl = DataLoader(test_ds, batch_size=1)

        return train_dl, test_dl, keys

    else:
        train_ds = Dataset(data=train_files, transform=train_transformer)
        train_dl = DataLoader(train_ds, batch_size=1)
        test_ds = Dataset(data=test_files, transform=test_transformer)
        test_dl = DataLoader(test_ds, batch_size=1)

        return train_dl, test_dl, keys

def dice_loss(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

def train_step(model, train_dl, keys, loss_fn, optim, device):
    total_train_loss = 0
    total_train_metric = 0
    step_count = 0
    for batch_data in tqdm(train_dl):
        step_count += 1
        volume = batch_data[keys[0]]
        label = batch_data[keys[1]]
        label = label != 0
        volume, label = (volume.to(device), label.to(device))
        optim.zero_grad()
        outputs = model(volume)    
        train_loss = loss_fn(outputs, label)   
        train_loss.backward()
        optim.step()

        total_train_loss += train_loss.item()
        train_metric = dice_loss(outputs, label)
        total_train_metric += train_metric
    total_train_loss /= step_count
    total_train_metric /= step_count
    return total_train_loss, total_train_metric

def test_step(model, test_dl, keys, loss_fn, optim, device):
    model.eval()
    with torch.inference_mode():
        total_test_loss = 0
        total_test_metric = 0
        step_count = 0

        for test_data in tqdm(test_dl):
            step_count += 1
            test_volume = test_data[keys[0]]
            test_label = test_data[keys[1]]
            test_label = test_label != 0
            test_volume, test_label = (test_volume.to(device), test_label.to(device),)
            test_outputs = model(test_volume)
            test_loss = loss_fn(test_outputs, test_label)
            total_test_loss += test_loss.item()
            test_metric = dice_loss(test_outputs, test_label)
            total_test_metric += test_metric
                    
        total_test_loss /= step_count
        total_test_metric /= step_count
        return total_test_loss, total_test_metric
    

def segmentation_training(model, input_data, loss, optim, max_epochs, save_dir, name, test_interval=1 , device=torch.device("cuda:0")):
    best_metric = 0
    best_epoch = 0
    
    results = {
        'train_loss': [],
        'train_metrics': [],
        'test_loss': [],
        'test_metrics': [],
        'epoch_values': []
    }
    path1 = os.path.join(save_dir, name + ".pt")
    path2 = os.path.join(save_dir, name + "_dict.pt")

    train_dl, test_dl, keys = input_data

    for epoch in tqdm(range(1,max_epochs)):
        train_loss, train_metric = train_step(model, train_dl, keys, loss, optim, device)
        test_loss, test_metric = test_step(model, test_dl, keys, loss, optim, device)
        print(f"Epoch: {epoch} | Train Loss: {train_loss: .5f} | Train Metric: {train_metric: .5f} | Test Loss: {test_loss: .5f} | Test Metric: {test_metric: .5f}")
        print("\n")
        results['epoch_values'].append(epoch)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['train_metrics'].append(train_metric)
        results['test_metrics'].append(test_metric)
        np.save(os.path.join(save_dir, 'epoch_values.npy'), results['epoch_values'])
        np.save(os.path.join(save_dir, 'train_loss.npy'), results['train_loss'])
        np.save(os.path.join(save_dir, 'test_loss.npy'), results['test_loss'])
        np.save(os.path.join(save_dir, 'train_metric.npy'), results['train_metrics'])
        np.save(os.path.join(save_dir, 'test_metric.npy'), results['test_metrics'])
        if test_metric > best_metric:
            torch.save(model, path1)
            torch.save(model.state_dict(), path2)
            best_metric = test_metric
            best_epoch = epoch
        if epoch >= 2:
            ptplot.segmentation_metric_plot(results['train_metrics'], results['train_loss'],
                                            results['test_metrics'], results['test_loss'], results['epoch_values'], save_dir)

    print(f"train completed, best_metric: {best_metric:.5f} "f"at epoch: {best_epoch}")
    
def segmentation_train_only(model, input_data, loss, optim, max_epochs, save_dir, name, device=torch.device("cuda:0")):
    best_metric = 0
    best_epoch = 0
    
    results = {
        'train_loss': [],
        'train_metrics': [],
        'epoch_values': []
    }
    path1 = os.path.join(save_dir, name + ".pt")
    path2 = os.path.join(save_dir, name + "_dict.pt")

    train_dl, test_dl, keys = input_data

    for epoch in tqdm(range(1,max_epochs)):
        train_loss, train_metric = train_step(model, train_dl, keys, loss, optim, device)
        print(f"Epoch: {epoch} | Train Loss: {train_loss: .5f} | Train Metric: {train_metric: .5f}")
        print("\n")
        results['epoch_values'].append(epoch)
        results['train_loss'].append(train_loss)
        results['train_metrics'].append(train_metric)
        np.save(os.path.join(save_dir, 'epoch_values.npy'), results['epoch_values'])
        np.save(os.path.join(save_dir, 'train_loss.npy'), results['train_loss'])
        np.save(os.path.join(save_dir, 'train_metric.npy'), results['train_metrics'])
        if train_metric > best_metric:
            torch.save(model, path1)
            torch.save(model.state_dict(), path2)
            best_metric = train_metric
            best_epoch = epoch
        if epoch >= 2:
            ptplot.segmentation_train_plot(results['train_metrics'], results['train_loss'], results['epoch_values'], save_dir)

    print(f"train completed, best_metric: {best_metric:.5f} "f"at epoch: {best_epoch}")

















