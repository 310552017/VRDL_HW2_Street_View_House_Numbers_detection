# VRDL_HW2_Street_View_House_Numbers_detection
SVHN dataset contains 33,402 trianing images, 13,068 test images
Train a not only accurate but fast digit detector!

## Coding Environment
- Window 10
- Jupyter Notebook
- Google Colab

## Reproducing Submission
To reproduct the testing prediction, please follow the steps below:
1. [Jupyter Notebook environment](#environment)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Testing](#testing)

## Environment
Jupyter_Notebook_environment.txt contains all packages version of Jupyter Notebook
- Python 3.8.0

## Dataset
- Because of using pytorch to predict the image, imagefolder could be used to generate training and testing data.
- Imagefolder needs the image be arranged in the folder according to their labels, so we need to classify the image with folders.
1. Download the dataset from https://competitions.codalab.org/my/datasets/download/83f7141a-641e-4e32-8d0c-42b482457836.
2. Create folder "data" .
3. Create folder "images" in "data"put the training image into "training_images" folder.
4. Create folder "training_images"、"training_labeled_images"、"testing_images"、"testing_labeled_images" in "images".
5. Extract the training image into "training_images" folder.
6. Extract the testing image into "testing_images" folder.
7. Then run the "image_label_classification.ipynb" code will classify the image according to their labels.
8. There are some examples of the folder in this porject above.


## Training
Upload the "310552017_Adjust_ResNet152.ipynb" and the classfied dataset file which is classify above to colab with google drive and running "310552017_Adjust_ResNet152.ipynb" file to train the model.

Remember to replace the root of the image file with your root.

The training parameters are:

Model | learning rate | Image size | Training Epochs | Batch size | optimizer
------------ | ------------- | ------------- | ------------- | ------------- | -------------
resnet152 | 0.001 | 224 | 15 | 32 | SGD

## testing
Testing accuracy with 15 epochs could reach 63% after upload to codalab.

Testing predition will be recorded in the answer.txt file.

### Pretrained models
Pretrained resnet152 model which is provided by pytorch.

### Link of my trained model
https://drive.google.com/file/d/1iCOjVgxlylJZvYKv4O_5bALpdgkBAQ7_/view?usp=sharing

### Inference

Load the trained model parameters without retraining again.

"Adjust_resnet152.pth" needs to be upload into google drive according to your root.

Then run the code of "inference.py" could get the "answer.txt" which contains the result of my model.
