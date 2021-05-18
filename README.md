# Drivable area detection
This project implements driving area segmentation by applying transfer learning on our model. We have used Berkeley Driving Dataset(BDD 100K) and data we collected on site to train our model.     
![Ops](https://github.com/Qiqi-0810/Drivable_area/blob/dfb263d26f552bcdf4b89fa207f10fc6d83c41e5/demo.png)

We use labelme to mark the picture and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) to train and test our model. However, we modify [it](https://github.com/Qiqi-0810/Drivable_area/tree/main/mmsegmentation) a bit to satisfy our requirement.

The experiments are down on local Anaconda and Google colab pro (equiped with Google drive).

This repository contain the whole process from data collection to model training and test:

* Picture extraction from videos [extract_jpg](https://github.com/Qiqi-0810/Drivable_area/blob/8b6ab833d52937251272ddbb032f23f69f141aad/extract_jpg.ipynb)
* Convert json to dataset in batch [json_to_dataset](https://github.com/Qiqi-0810/Drivable_area/blob/8b6ab833d52937251272ddbb032f23f69f141aad/json_to_dataset.py).
It can be used in cmd as: python json_to_dataset.py E:\pathofyourjsondata
* Batch extract png files from dataset files to a new folder [copy_png_to](https://github.com/Qiqi-0810/Drivable_area/blob/8b6ab833d52937251272ddbb032f23f69f141aad/copy_png_to.ipynb)
* Generate txt files for training and validating (optional) [generate_txt](https://github.com/Qiqi-0810/Drivable_area/blob/8b6ab833d52937251272ddbb032f23f69f141aad/generate_txt.ipynb)
* Count the pixels and instance numbers of your dataset (optional, cause we use it to count class weight for imbalanced dataset) [countlabel](https://github.com/Qiqi-0810/Drivable_area/blob/8b6ab833d52937251272ddbb032f23f69f141aad/countlabel.ipynb)
* Calculate the mean and variance as normalization parameters （optional) [mean&std](https://github.com/Qiqi-0810/Drivable_area/blob/8b6ab833d52937251272ddbb032f23f69f141aad/mean&std.ipynb)
* Change label format to P mode if needed (Optional， It can also be used to change the bit depth of the image. It also includes putting the palette into a png tag file for coloring) [image_format](https://github.com/Qiqi-0810/Drivable_area/blob/8b6ab833d52937251272ddbb032f23f69f141aad/image_format.ipynb)

# Getting started
## Training, valisating and test
The whole process of training and upload required files are done in Colab files []()
## Dataset
Dataset can be downloaded from Berkeley Driving Dataset. There are 70k training images, 10k validation images and 20k test images. These should be copied in the following path
The whole dataset structure is like below:
```
├── data
│   ├── Qidataset
│   │   ├── images
│   │   │   ├── xxx.jpg
│   │   │   ├── yyy.jpg
│   │   ├── labels
│   │   │   ├── xxx.png
│   │   │   ├── yyy.png
│   │   ├── splits
│   │   │   ├── train480.txt
│   │   │   ├── train1200.txt
│   │   │   ├── train10000.txt
│   │   │   ├── val160.txt
│   │   │   ├── val400.txt
│   │   │   ├── val1000.txt
│   │   │   ├── test.txt
│   │   │   ├── Night_val.txt
│   │   │   ├── Day_val.txt
│   │   │   ├── Rain_val.txt
│   │   │   ├── Cloud_val.txt
│   │   │   ├── Sunny_val.txt
│   │   │   ├── Snow_val.txt
│   │   │   ├── Structured_val.txt
│   │   │   ├── unstructured_val.txt
```
## Results and analysis
Results can be analyzed by running notebooks in mmsegmentation/tools

# Tips
* If you are going to use [mmseg](https://mmsegmentation.readthedocs.io/en/latest/index.html) or [BDD100K](https://doc.bdd100k.com/usage.html), definitely read the tutorial carefully.
* Try not to write config files from scratch, modified it and import other files then change the part you want to change.
```
_base_='./dnl_r50-d8_512×1024_40k_cityscapes.py'
norm_cfg = dict(type = 'BN', requires_grad=True)
model = dict(
    backbone=dict(
        dilations=(1, 2, 5, 1),
        norm_cfg=norm_cfg))
```
* You can always import a packaged module from mmcv.cnn or mmseg to reduce the amount of code.
* Never forget to register the module you designed and import it in file `mmseg/models/decode_heads/_init_.py
