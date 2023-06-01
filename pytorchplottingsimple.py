import torch
from torchvision.utils import make_grid
import pytorchsimple as pts
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
import sys
import itertools
from torchvision.utils import save_image
from glob import glob
import cv2
from monai.inferers import sliding_window_inference
from monai.utils import first, set_determinism
from monai.transforms import Activations

def export_result(images, test_result_name, test_pred_name, path = os.getcwd()):
	
	df = pd.DataFrame(
    				{'Filename': images,
    				 'Actual Class': test_result_name,
     				 'Predicted Class': test_pred_name
    				})
	index = 2
	current = path
	x = input("Enter result csv filename: ")
	path = current + x + ".csv"

	if not os.path.exists(path):
		df.to_csv(path)
	else:
		new_path = path.split(".")
		tmp = new_path[0] + "{}".format(index)
		new_path = tmp + ".csv"
		df.to_csv(new_path)

def training_plot(history, result_path, type='loss'):
    epoch = history['epoch_values']
    if type=='a' or type == 'acc' or type == 'accuracy':
        train = history['train_acc']
        test = history['test_acc']
    elif type=='l' or type =='loss':
        train = history['train_loss']
        test = history['test_loss']
    plt.plot(epoch, train, label="train set")
    plt.plot(epoch, test, label="test set")
    plt.legend()
    plot_path = result_path + 'model_' + type + '.jpg'
    plt.savefig(plot_path)
    plt.show()
    
def cm_plot(result_path, val_result_name, val_pred_name, images=0):
    outputs = np.unique(val_result_name)
    df_confusion_matrix = confusion_matrix(val_result_name, val_pred_name, labels = outputs)
    result_path = result_path + 'model_cm_plot.jpg'
    title = result_path.split("/")
    title = title[-1]
	###To plot confusion matrix
    cmap = plt.cm.Blues

    plt.figure(figsize=(8,8))
    df_confusion_matrix = (df_confusion_matrix.astype('float')/df_confusion_matrix.sum(axis=1)[:,np.newaxis])*100
    plt.imshow(df_confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(outputs))
    plt.xticks(tick_marks, outputs)
    plt.yticks(tick_marks, outputs)

    fmt = '.2f'
    thresh = df_confusion_matrix.max() / 2.
    for i, j in itertools.product(range(df_confusion_matrix.shape[0]), range(df_confusion_matrix.shape[1])):
        plt.text(j, i, format(df_confusion_matrix[i, j], fmt), horizontalalignment="center", color="white" if df_confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(result_path)
    plt.show()
    if images == 0:
        sys.exit()
    else:
        export_result(images, val_result_name, val_pred_name)
        
def show_images(images, stats, nmax=64):
    fig, ax = plt.subplots(figsize = (8,8))
    ax.set_xticks([])
    ax.set_yticks([])
    denormed = pts.img_denorm(images.detach()[:nmax], stats)
    ax.imshow(make_grid(denormed, nrow=8).permute(1, 2, 0))
    plt.show()

def show_batch(dl, stats, nmax=64):
    for images, _ in dl:
        show_images(images, stats, nmax)
        break
    
def save_samples(generator, save_dir, index, latent_tensors, stats, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(pts.img_denorm(fake_images, stats), os.path.join(save_dir, fake_fname), nrow=8)
    print('Saving...', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images, nrow=8).permute(1,2,0))
        plt.show()

def segmentation_metric_plot(train_metric, train_loss, test_metric, test_loss, epochs, save_dir):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.title("Train loss")
    x = epochs
    plt.plot(x, train_loss)

    plt.subplot(2, 2, 2)
    plt.title("Train metric")
    plt.plot(x, train_metric)

    plt.subplot(2, 2, 3)
    plt.title("Test loss")
    plt.plot(x, test_loss)

    plt.subplot(2, 2, 4)
    plt.title("Test metric")
    plt.plot(x, test_metric)

    plt.savefig(os.path.join(save_dir, 'train_test_plots.jpg'))

def segmentation_train_plot(train_metric, train_loss, epochs, save_dir):
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    plt.title("Train loss")
    x = epochs
    plt.plot(x, train_loss)

    plt.subplot(1, 2, 2)
    plt.title("Train metric")
    plt.plot(x, train_metric)
    
    plt.savefig(os.path.join(save_dir, 'train_test_plots.jpg'))

def video_from_img(imgfolder, img_shape, vid_name, output_dir):
    shape = img_shape
    out = cv2.VideoWriter(os.path.join(output_dir, vid_name+'.mp4'), 
                          cv2.VideoWriter_fourcc(*'mp4v'), 8, (shape[0], shape[1]))
        
    files = glob(imgfolder+'/*.jpg')
    sfiles = sorted(files, key=lambda t: os.stat(t).st_mtime)
    for file in sfiles:
        img = cv2.imread(file)
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        out.write(img)
    out.release

def sample_img_creator(model, data_loader, imgfolder, train_loader=False, device=torch.device('cpu')):
    model.eval()
    sw_batch_size = 4
    roi_size = (128, 128, 64)

    with torch.inference_mode():
        test = first(data_loader)
        t_volume = test['volume']
        
        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs = sigmoid_activation(test_outputs)
        test_outputs = test_outputs > 0.53
        if train_loader:
            p = 3
        else:
            p = 2
        for i in range(0, 64):
            plt.figure("check", (18, 6))
            plt.subplot(1, p, 1)
            plt.axis('off')
            plt.title(f"Image {i}")
            plt.imshow(test["volume"][0, 0, :, :, i], cmap="gray")
            # print(test_patient["volume"][0, 0, :, :, i].shape)
            if p == 3:
                plt.subplot(1, p, 2)
                plt.axis('off')
                plt.title(f"Label {i}")
                plt.imshow(test["segment"][0, 0, :, :, i] != 0)
            plt.subplot(1, p, p)
            plt.axis('off')
            plt.title(f"output {i}")
            plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
            plt.savefig(os.path.join(imgfolder, 'Image{}.jpg'.format(i)))
