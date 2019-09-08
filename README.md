# Mask R-CNN for Cookies Detection and Segmentation

This is a tiny project to use Mask R-CNN for detecting two brands of cookies **"Lays"** and **"Doritos"**.  
Most of the code is based on the implementation of [Mask R-CNN by matterport](https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow. Where we modified is change the backbone network from ResNet-101 to ResNet-50 and the batch size from 2 to 1 image. This setting is will use 97~98% memory of NVIDIA RTX2060 6GB.  


![Instance Segmentation Sample](assets/detect_10509506623156.jpg)

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet-50.
* Instruction and training code for the toy dataset of 2 kinds of cookies.
* Pre-trained weights on MS COCO and ImageNet.
* Jupyter notebooks to visualize the detection pipeline at every step
* Example of training on your own dataset

If you just want to try the detection, you can download my trained model [here](https://drive.google.com/file/d/18BOn-qlodw1oebFRQk5P0ZHlcc83jHMt/view?usp=sharing) and jump to **Detection** part. 


# Training your dataset
First, the pre-trained weights from MS COCO and ImageNet are also provided for users to fine-tune their own datasets. You can start from reading the tutotial on this [blog](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) which introduce the whole process from collecting images, labelling, and training. So in this document, we're going to introduce the results of our toy dataset rather than describe too many details.

Note that the data directory strutures are as following, and besure to named your json file correctly:  
```
datasets  
    └ cookies
        └ train
            └ img1.jpg
            └ img2.jpg
            └ ...
            └ via_region_data_cookies_train.json
        └ val
            └ val_img1.jpg
            └ val_img2.jpg
            └ ...
            └ via_region_data_cookies_val.json
        └ test
            └ test_img1.jpg
            └ test_img2.jpg
            └ ...
```

To start training on your own dataset, the things we need to chang in this repo is:  

* ``CookiesConfig`` in [cookies.py](/cookies.py):  
    The change here is to set the appropriate parameters on training or using your GPU memory as mention above. If you have better GPU with larger memory, you can increase the setting like ``IMAGES_PER_GPU``, or change the ``BACKBONE`` to ``"resnet101"``.
* ``CookiesDataset`` in [cookies.py](/cookies.py):  
    Here you need to add the class of your datasets like  
    ```
    self.add_class("cookies", 1, "lays")  
    self.add_class("cookies", 2, "doritos")  
    .
    .
    .
    ```  
    And remenber to collect the ``class_names`` (in line 164, 167, 182) in your .json annotation file from [VIA tool](http://www.robots.ox.ac.uk/~vgg/software/via/via-2.0.7.html).

If you set those thing above ready, be sure to use the jupyter notebook to run the notebook [inspect_cookies_data.ipynb](/notebook/inspect_cookies_data.ipynb) to check the data is going to be read and process correctly.  
Then you can start training your model by following command. 
```
# remenber to pass the --logs to save the weight file (.h5) and the logs during training in tensorboard:
python3 cookies.py train --dataset=/path/to/cookies/dataset --weights=imagenet --logs==/path/to/log  
# with augmentation:
python3 cookies.py train --dataset=/path/to/cookies/dataset --weights=imagenet --augmentation=1 --logs=/path/to/log
```
Here we use pre-trained model from imagenet is to let us not need to install some other packages to support with COCO.


# Monitoring your training:
You can not only use tensorboard to see the logs during training, but also can use the [inspect_cookies_model.ipynb](/notebook/inspect_cookies_model.ipynb) to see the effect of the current trained model, or [inspect_cookies_weights.ipynb](/notebook/inspect_cookies_weights.ipynb) to visualize the stats and distribution of the weights in current model.  
Those files are set to run on CPU which can let you keep training on GPU while using this notebook. And it also provide step-by-step detection to see the detail on each stages in detecting with the current model.

# Detection
To detecting, there are two ways you can do:
1. Use [demo_cookies_detection.ipynb](/notebook/demo_cookies_detection.ipynb) with CPU.
2. Run the following commands on GPU:
```
python3 cookies.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>
```
If you are using way 1., I also added the parameter of ``savedir`` in ``display_instances()`` in [visualize.py](./mrcnn/visualize.py) to let user save the detection results easier.
But I still implemented the new sub-function ``draw_on_image``  in [cookies.py](/cookies.py) to draw the bounding box and mask on the image directly which is used in the detection process. You can also try this function in [demo_cookies_detection.ipynb](/notebook/demo_cookies_detection.ipynb).

# Instance Segmentation Samples on Cookies Dataset
![](assets/detect_10506228923128.jpg)  

![](assets/detect_10506227236894.jpg)  

![](assets/detect_google_0484.jpg)  

![](assets/detect_MT190319164316509218.jpg)