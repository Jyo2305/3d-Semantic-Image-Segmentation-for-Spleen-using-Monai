# 3d-Image-Segmentation-for-Spleen-using-Monai
<br>
original dataset: https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu/view?usp=drive_link (from Medical Segmentation Decathlon Datasets)<br><br>

## **Testing on a labeled data:**<br>
![train_img](https://github.com/Jyo2305/3d-Semantic-Image-Segmentation-for-Spleen-using-Monai/blob/main/results/spleen_segment/train_gif.gif)<br>

## File descriptions:<br>
spleen_segmentation - This notebook file contains the steps to train a model for 3d semantic segmentation and it is here tested on 'Spleen Segmentation'. The notebook can be used for any other 3d semantic segmentation as long as the data has been prepared properly (i.e. has the same number of slices per
volume and labels file)<br><br>
segmentation_utils - It contains all the required helper functions for the data loading and model training<br><br>
pytorchplottingsimple - It contains all the required helper functions for plotting the data<br><br>
pytorchsimple - It contains helper functions for model building and 2d image classification and GANs (not implemented in this project)<br><br>
results/spleen_segment - This folder contains the results of the training and the model<br><br>
## Requirements<br>
The versions of packages used here is mentioned in brackets:<br>
Matplotlib (3.7.1)<br>
Torch (2.0.0)<br>
Torchvision (0.15.1)<br>
Numpy (1.24.2)<br>
Torchinfo (1.8.0)<br>
tqddm (4.65.0)<br>
Monai (1.1.0)<br>
Glob2 (0.7)<br>
opencv-python (4.7.0.72)<br>
scikit-learn (1.2.2)<br>


