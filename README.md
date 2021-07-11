## Table of Contents
- [Intro](#intro)
- [Marked up WiderFace](#marked-up-widerface)
- [Gender Classification](#gender-classification)
  * [EfficientNet model](#efficientnet-model)
  * [Results on Gendered WiderFace](#results-on-gendered-widerface)
  * [Results on forensic set](#results-on-forensic-set)
  * [Inference Latency](#inference-latency)
  * [ROC-AUC on forensic set](#roc-auc-on-forensic-set)
- [Usage](#usage)
  * [Face Detector](#face-detector)
  * [Python script](#python-script)
  * [.NET WPF solution](#net-wpf-solution)
    + [Convert pytorch model to onnx](#convert-pytorch-model-to-onnx)
    + [How to use](#how-to-use)
- [Paper](#paper)
- [References](#references)

## Intro
Gender classification algorithm in the forensic field based on [EfficientNet](https://github.com/qubvel/efficientnet) model and manually marked up by gender [WiderFace](http://shuoyang1213.me/WIDERFACE/). 

## Marked up WiderFace
WiderFace was selected because the images in this set are as close as possible to the forensic photos. This set is "more wild" (extreme scale, occlusion, etc) than similar datasets such as [FairFace](https://github.com/joojs/fairface), [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html), [IMDb-Wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). 
After we take easy-scale part of images and marked up them by gender.
Total amount of faces in our data set -- 12 662 (6331 Males and 6331 Females) . 
The dataset is available on [Kaggle](https://www.kaggle.com/fedoszhilkin/genderwiderface) or [Drive](https://drive.google.com/file/d/1YwhdGujhdelwhqTAXzC6TYJwoPY0k65v/view?usp=sharing).

## Gender Classification 
### EfficientNet model
We chose EfficientNet model pretrained on Imagenet for training on Our data set. 
### Results on Gendered WiderFace 
| Method          | Trained on | Precision | Recall  | F1    |
|-----------------|------------|-----------|---------|-------|
| EfficientNet-B0 | Our        | 0.962     |0.97   |0.966|
| EfficientNet-B2 | Our        | 0.978     |0.97   | 0.974 |
| **EfficientNet-B4** | **Our**        | **0.974**     | **0.974**  | **0.974**   |
| Gil Levi model  | Adience    |0.745     | 0.735    |0.740|
| VGG-Face        | IMDb-WIKI  |0.581     | 0.97   | 0.727|
| FairFace        | FairFace   |0.773    | 0.927   | 0.843 |
### Results on forensic set
| Method          | Trained on | Precision | Recall  | F1    |
|-----------------|------------|-----------|---------|-------|
| EfficientNet-B0 | Our        | 0.924     | 0.872   | 0.897 |
| EfficientNet-B2 | Our        | 0.911     | 0.887   | 0.899 |
| **EfficientNet-B4** | **Our**        | **0.902**     | **0.897**  | **0.9**   |
| Gil Levi model  | Adience    | 0.893     | 0.6     | 0.718 |
| VGG-Face        | IMDb-WIKI  | 0.649     | 0.985   | 0.782 |
| FairFace        | FairFace   | 0.829     | 0.897   | 0.862 |

### Inference Latency
| Method          | Min time | Max time | Avg time |
|-----------------|----------|----------|----------|
| EfficientNet-B0 | 0.24     | 0.39     | 0.28     |
| EfficientNet-B2 | 0.39     | 1.2      | 0.51     |
| EfficientNet-B4 | 0.76     | 1.66     | 0.87     |
| **Gil Levi model**  |**0.12**     |**0.24**     | **0.13**     |
| VGG-Face        | 1.57     | 2.87     | 1.75     |
| FairFace        | 0.87     | 1.61     | 0.94     |

### ROC-AUC on forensic set
![изображение](https://user-images.githubusercontent.com/23313519/117559598-59e62900-b08f-11eb-8346-7c58ae12f1c2.png)

## Usage
### Face Detector
We use [CenterFace](https://github.com/Star-Clouds/CenterFace) detector for python and C# projects.
- [Python-proj face model](https://github.com/Feodoros/ForensicGenderSex/tree/master/PretrainedModels/FaceDetecting)
- [C#-proj face model](https://github.com/Feodoros/ForensicGenderSex/tree/master/PredictGenderWPF/Models)

### Python script
You can download our pretrained gender prediction models from [here](https://github.com/Feodoros/ForensicGenderSex/tree/master/PretrainedModels/GenderEstimation/Ours).
Usage of [this](https://github.com/Feodoros/ForensicGenderSex/blob/master/predict_gender.py)  python script:

``predict_gender.py -i Input image path -o Output image path -d Detect faces True/False (default: True) ``

### .NET WPF solution
You can test gender prediction (EfficientNet-B0 model) with windows desktop [app](https://github.com/Feodoros/ForensicGenderSex/tree/master/PredictGenderWPF).
#### Convert pytorch model to onnx
We should convert our pretrained model to onnx format to use this model in .NET with [onnxruntime](https://github.com/microsoft/onnxruntime).
Use this [notebook](https://github.com/Feodoros/ForensicGenderSex/blob/master/pytorch2onnx.ipynb) to convert to onnx.
You can download prepaired onnx model from [here](https://github.com/Feodoros/ForensicGenderSex/tree/master/PredictGenderWPF/Models). 
#### How to use
+ Open image
+ Click Analyze 
+ Click on face on image or select this face from list to see this face closer
+ Right click on image to save finished image

![изображение](https://user-images.githubusercontent.com/23313519/117559532-d6c4d300-b08e-11eb-9a9d-be35f7d9cc18.png)

## Paper

## References
+ [EfficientNet](https://github.com/qubvel/efficientnet)
+ [WiderFace](http://shuoyang1213.me/WIDERFACE/)
+ [CenterFace](https://github.com/Star-Clouds/CenterFace)
+ [onnxruntime](https://github.com/microsoft/onnxruntime)

