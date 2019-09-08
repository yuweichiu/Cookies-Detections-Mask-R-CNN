"""
Mask R-CNN
Train on the toy cookies dataset to recognize 'lays' and 'doritos'.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cookies.py train --dataset=/path/to/cookies/dataset --weights=coco

    # Resume training a model that you had trained earlier and specify the directory of training logs
    python3 cookies.py train --dataset=/path/to/cookies/dataset --weights=last --logs==/path/to/log

    # Train a new model starting from ImageNet weights and augmentation
    python3 cookies.py train --dataset=/path/to/cookies/dataset --weights=imagenet --augmentation=1 --logs=/path/to/log

    # Detect on an image:
    python3 cookies.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Detect on an image:
    python3 cookies.py detect --weights=/path/to/weights/file.h5 --video=<URL or path to file>

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug.augmenters as iaa
import cv2

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Session Setting
############################################################
# If you face the error about convolution layer,
# use this block to enable the memory usage of GPU growth.
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

############################################################
#  Configurations
############################################################


class CookiesConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cookies"

    # I use a NVIDIA RTX2060 6GB , which can fit 1 images with ResNet-50
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + lays + doritos

    # Backbone network of Mask R-CNN:
    # This option can change between ResNet-101 and ResNet-50
    # up to the capacity of your GPU.
    BACKBONE = "resnet50"

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 115

    # Skip detections with < 95% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8

    # Class name list:
    CLASS_NAME_LIST = ['BG', 'lays', 'doritos']

    # Detection visualization options for OpenCV:
    VISUALIZE_FONT = cv2.FONT_HERSHEY_DUPLEX
    VISUALIZE_FONT_SCALE = 1
    VISUALIZE_FONT_THICKNESS = 1
    VISUALIZE_MASK_TRANSPARENCY = 0.5


############################################################
#  Dataset
############################################################

class CookiesDataset(utils.Dataset):

    def load_cookies(self, dataset_dir, subset):
        """Load a subset of the Cookies dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.onv.
        self.add_class("cookies", 1, "lays")
        self.add_class("cookies", 2, "doritos")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data_cookies_" + subset + ".json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                class_name = [r['region_attributes']['cookies'] for r in a['regions']]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                class_name = [r['region_attributes']['cookies'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "cookies",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_name=class_name
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cookies":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_id = []
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            class_name = info['class_name'][i]
            class_id.append(self.class_names.index(class_name))

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.array(class_id, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cookies":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, epoch, augmentation=False):
    """Train the model."""
    # Training dataset.
    dataset_train = CookiesDataset()
    dataset_train.load_cookies(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CookiesDataset()
    dataset_val.load_cookies(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epoch,
                augmentation=augmentation,
                layers='heads')


def draw_on_image(image, r, colors, bbox=True, mask=True, score=True):
    """
    Draw the bbox, mask on image with OpenCV.
    :param image: opencv-based BGR image matrix.
    :param r: detection result.
    :param colors: the color list for each object.
    :param bbox: detection results ['rois']
    :param score: detection results ['score']
    :param mask: detection results ['mask']
    :param color_per_class: put color on each class or not.
            [c1, c2, ..., cN] for specifying those N class with color c1~cN.
            None will apply a random colors list.
    :return: Drawn image
    """
    new_img = image.copy()
    cookies_config = CookiesConfig
    class_names = cookies_config.CLASS_NAME_LIST
    font = cookies_config.VISUALIZE_FONT
    font_scale = cookies_config.VISUALIZE_FONT_SCALE
    thickness = cookies_config.VISUALIZE_FONT_THICKNESS
    alpha = cookies_config.VISUALIZE_MASK_TRANSPARENCY
    cBGR, cBGR255 = [], []

    for rid in range(len(r['rois'])):
        bbox1 = r['rois'][rid]
        mask1 = r['masks'][:, :, rid]
        pt1, pt2 = (bbox1[1], bbox1[0]), (bbox1[3], bbox1[2])
        cBGR.append(colors[rid][::-1])  # color RGB to BGR
        color255 = tuple([255 * i for i in colors[rid]])  # scale to 0 ~ 255
        cBGR255.append(color255[::-1])
        if bbox:
            new_img = cv2.rectangle(image, pt1, pt2, cBGR255[-1], 2)
        if mask:
            new_img = visualize.apply_mask(new_img, mask1, cBGR[-1], alpha)
    if score:
        # Finally put text on each target with text box:
        for rid in range(len(r['rois'])):
            bbox1 = r['rois'][rid]
            text_x, text_y = bbox1[1], bbox1[0]
            pt1 = (text_x, text_y)
            text = class_names[r['class_ids'][rid]] + ' ' + '{0:.3f}'.format(r['scores'][rid])
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
            # Set the text box position:
            tbx_p1_x, tbx_p1_y = bbox1[1] - 5, bbox1[0] - text_height - 5
            tbx_p2_x, tbx_p2_y = bbox1[1] + text_width + 5, bbox1[0] + 5
            if tbx_p2_x > new_img.shape[1]:
                text_x -= tbx_p2_x - new_img.shape[1]
                tbx_p1_x -= tbx_p2_x - new_img.shape[1]
                tbx_p2_x -= tbx_p2_x - new_img.shape[1]
            if tbx_p1_y < 0:
                text_y -= tbx_p1_y
                tbx_p1_y -= tbx_p1_y
                tbx_p2_y = tbx_p1_y + text_height + 10
            tbx_p1 = (tbx_p1_x, tbx_p1_y)
            tbx_p2 = (tbx_p2_x, tbx_p2_y)
            pt1 = (text_x, text_y)
            new_img = cv2.rectangle(new_img, tbx_p1, tbx_p2, cBGR255[rid], cv2.FILLED)
            new_img = cv2.putText(new_img, text, pt1,
                                  font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return new_img


def detect(model, weights_path, image_path=None, video_path=None, colors_each_class=None, show=True):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        if not colors_each_class:
            colors = visualize.random_colors(len(r['class_ids']))
        else:
            colors = [colors_each_class[cid] for cid in r['class_ids']]
        # Convert RGB to BGR in opencv:
        image_cv = image[:, :, ::-1]
        image_cv = image_cv.copy()
        # Draw on image:
        new_img = draw_on_image(image_cv, r, colors=colors)
        out_dir = weights_path.split(os.path.basename(weights_path))[0]
        file_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(out_dir, 'detect_' + file_name), new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print("Saved to ", file_name)
        if show:
            cv2.imshow('', new_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return new_img
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "detection_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image_cv = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image_cv[..., ::-1].copy()
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                colors = [colors_each_class[cid] for cid in r['class_ids']]
                # Draw on image:
                new_img = draw_on_image(image_cv, r, colors=colors)
                # Add image to video writer
                vwriter.write(new_img)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cookies.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cookies/dataset/",
                        help='Directory of the Cookies dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--augmentation', required=False,
                        default=0,
                        metavar="Data augmentation or not",
                        help='Use augmentation or not')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to detect')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CookiesConfig()
    else:
        class InferenceConfig(CookiesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Augmentation options:
        if args.augmentation.lower() == '1':
            augmentation = iaa.SomeOf((1, 2), [
                iaa.Affine(rotate=[-20, 20]),
                iaa.Rot90(1, True),
                iaa.Rot90(3, True),
                iaa.Rot90(2, True),
                iaa.ChannelShuffle(0.5),
                iaa.PerspectiveTransform(0.075)
            ])
        else:
            augmentation = False
        train(model, epoch=300, augmentation=augmentation)
    elif args.command == "detect":
        cookies_config = CookiesConfig
        class_names = cookies_config.CLASS_NAME_LIST
        colors = visualize.random_colors(len(class_names))
        detect(model, weights_path, image_path=args.image, video_path=args.video, colors_each_class=colors)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
