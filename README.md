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
- “Transfer_mat_To_csv.ipynb” which can transfer .mat file to .csv file. 
- I transfer the file on google colab and save on the google drive , then download the csv file to my computer to train the model.
1. Download the dataset from https://drive.google.com/drive/folders/1aRWnNvirWHXXXpPPfcWlHQuzGJdXagoc.
2. Upload file "digitStruct.mat"、"Transfer_mat_To_csv.ipynb" to google drive in the same folder.
3. Run the file "Transfer_mat_To_csv.ipynb" you will get a "train_ann.csv" file.
4. Download the file "train_ann.csv".
5. Extract the "train.zip" and "test.zip" in the folder with "train_ann.csv".
6. Training images will be put in "train" folder and testing images will be put in "test" folder.


## Training
- Download the files "VRDL_HW2_train.ipynb"、“utils.py”、“transforms.py”、“coco_eval.py”、“model_utils.py”、“engine.py”、“coco_utils.py” and put these files in the folder with - "train_ann.csv"
- Run the files "VRDL_HW2_train.ipynb" will start to train the model and save it.
- Remember to replace the root of the image file with your own root.

The training parameters are:

Model | learning rate | Training Epochs | Batch size | optimizer
------------------------ | ------------------------- | ------------------------- | ------------------------- | -------------------------
FasterRCNN_resnet50_fpn | 0.005 | 5、10 | 4 | SGD

## testing
- "VRDL_HW2_train.ipynb" has the code that can use the model which is saved above to predict the testing images and save the prediction result as json files according to coco set rules.

### Pretrained models
Pretrained model "fasterrcnn_resnet50_fpn" which is provided by torchvision.

### Link of my trained model
////https://drive.google.com/file/d/1iCOjVgxlylJZvYKv4O_5bALpdgkBAQ7_/view?usp=sharing

### Inference

Load the trained model parameters without retraining again.

".pth" needs to be download to your own device and run "VRDL_HW2_train.ipynb" you will get the results as json file.

"Inference.ipynb" just has the code about calculating the running time of the model and "Inference.ipynb" and the model need to be upload to google colab to run so that it can has the same hard device.
