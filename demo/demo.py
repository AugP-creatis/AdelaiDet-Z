# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

from detectron2.data import DatasetCatalog, MetadataCatalog
import random
import json
from detectron2.structures import BoxMode

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TEST = ('test',)

    # STACK
    #cfg.INPUT.IS_STACK = True
    #cfg.INPUT.STACK_SIZE = 11
    #cfg.INPUT.EXTENSION = ".png"
    #cfg.INPUT.SLICE_SEPARATOR = "F"

    # INPUT
    #cfg.INPUT.FORMAT = "BGR"   # Model input format (has to be BGR for seamless inference when reading RGB images with opencv)
    #cfg.INPUT.MASK_FORMAT = "polygon"
    #cfg.INPUT.MIN_SIZE_TRAIN = (480,)
    #cfg.INPUT.MAX_SIZE_TRAIN = 1333
    #cfg.INPUT.MIN_SIZE_TEST = 480
    #cfg.INPUT.MAX_SIZE_TEST = 1333

    # MODEL
    #cfg.MODEL.WEIGHTS = ""
    #cfg.MODEL.BACKBONE.FREEZE_AT = 0

    cfg.MODEL.EARLY_FILTER.ENABLED = True
    cfg.MODEL.EARLY_FILTER.OPERATOR = "Sobel"
    
    #cfg.MODEL.USE_AMP = True
    #cfg.MODEL.META_ARCHITECTURE = "CondInst_Z"
    #cfg.MODEL.BACKBONE.DIM = 3
    cfg.MODEL.BACKBONE.INTER_SLICE = True
    #cfg.MODEL.BACKBONE.ANTI_ALIAS = False
    #cfg.MODEL.RESNETS.DEFORM_INTERVAL = 1
    #cfg.MODEL.MOBILENET = False
    #cfg.MODEL.RESNETS.DEPTH = 18
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 32
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = {10:16, 18:64, 32:64, 50:256, 101:256, 152:256}[cfg.MODEL.RESNETS.DEPTH]
    cfg.MODEL.FCOS.SIZES_OF_INTEREST = [32, 64, 128, 256] if cfg.MODEL.RESNETS.DEPTH == 10 else [64, 128, 256, 512]
    #cfg.MODEL.RESNETS.NORM = "BN3d"
    #cfg.MODEL.RESNETS.RES5_DILATION = 1
    #cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True
    cfg.MODEL.FPN.OUT_CHANNELS = 64
    #cfg.MODEL.SEPARATOR.NAME = "From3dTo2d"

    cfg.MODEL.FCOS.NUM_CLS_CONVS = 1
    cfg.MODEL.FCOS.NUM_BOX_CONVS = 1
    cfg.MODEL.FCOS.NUM_CLASSES = len(eval(args.classes_dict))  #For FCOS and CondInst
    #cfg.MODEL.MEInst.NUM_CLASSES = len(eval(args.classes_dict)) #For MeInst

    cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 1
    cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 32
    cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
    cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 2
    cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS = 4

    #cfg.MODEL.PIXEL_MEAN = [87.779, 100.134, 101.969]   #In BGR order
    #cfg.MODEL.PIXEL_STD = [16.368, 13.607, 13.170]  #In BGR order

    cfg.OUTPUT.FILTER_DUPLICATES = False
    cfg.OUTPUT.GATHER_STACK_RESULTS = False

    #Remove all online augmentations
    #cfg.INPUT.HFLIP_TRAIN = False
    #cfg.INPUT.CROP.ENABLED = False
    #cfg.INPUT.IS_ROTATE = False
    #cfg.TEST.AUG.ENABLED = False

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")

    parser.add_argument('--data-dir', default='/home/perrier/Bacteriocytes_seg/data')
    parser.add_argument('--classes-dict',type=str,default="{'Intact_Sharp':0, 'Broken_Sharp':2}")
    #Classes are like "{'Intact_Sharp':0,'Intact_Blurry':1,'Broken_Sharp':2,'Broken_Blurry':3}"
    parser.add_argument('--cross-val', default=0)

    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

# get_dicts from Nathan Hutin https://gitlab.in2p3.fr/nathan.hutin/detectron2/-/blob/main/train_cross_validation.py
# inspired from official Detectron2 tutorial notebook https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

def get_dicts(dir, mode, idx_cross_val, classes):
    """
    Read the annotations for the dataset in YOLO format and create a list of dictionaries containing information for each
    image.

    Args:
        img_dir (str): Directory containing the images.
        ann_dir (str): Directory containing the annotations.

    Returns:
        list[dict]: A list of dictionaries containing information for each image. Each dictionary has the following keys:
            - file_name: The path to the image file.
            - image_id: The unique identifier for the image.
            - height: The height of the image in pixels.
            - width: The width of the image in pixels.
            - annotations: A list of dictionaries, one for each object in the image, containing the following keys:
                - bbox: A list of four integers [x0, y0, w, h] representing the bounding box of the object in the image,
                        where (x0, y0) is the top-left corner and (w, h) are the width and height of the bounding box,
                        respectively.
                - bbox_mode: A constant from the `BoxMode` class indicating the format of the bounding box coordinates
                             (e.g., `BoxMode.XYWH_ABS` for absolute coordinates in the format [x0, y0, w, h]).
                - category_id: The integer ID of the object's class.
    """
    random.seed(0)
    if mode == 'train':
        cross_val_dict = {0:[2,3,4], 1:[0,3,4], 2:[0,1,4], 3:[0,1,2], 4:[1,2,3]}
        folds_list = cross_val_dict[idx_cross_val]

    elif mode == 'val' :
        cross_val_dict = {0:[1], 1:[2], 2:[3], 3:[4], 4:[0]}
        folds_list = cross_val_dict[idx_cross_val]
    
    else:
        cross_val_dict = {0:[0], 1:[1], 2:[2], 3:[3], 4:[4]}
        folds_list = cross_val_dict[idx_cross_val]

    dataset_dicts = []
    dict_instance_label = {value:num for num, value in enumerate(classes.values())}
    for fold in folds_list:
        img_dir = os.path.join(dir, 'Cross-valRGB', 'Xval'+str(fold)+'_images', 'images')
        ann_dir = os.path.join(dir, 'Cross-valRGB', 'Xval'+str(fold)+'_labels','detectron2')
    

        for idx, file in tqdm(enumerate(os.listdir(ann_dir)), desc=f'cross validation {fold}, mode {mode}'):
            # annotations should be provided in yolo format
            if mode !='train' and 'Augmented' in file:
                continue

            record = {}
            dico = json.load(open(os.path.join(ann_dir, file)))

            record["file_name"] = os.path.join(img_dir, dico['info']['filename'])
            record["image_id"] = dico['info']['image_id']
            record["height"] = dico['info']['height']
            record["width"] = dico['info']['width']

            objs = []
            for instance in dico['annotation']:
                if 'Trash' in classes.keys() and instance['category_id'] in classes['Trash']:
                    instance['category_id'] = 1

                if instance['category_id'] in classes.values() or ('trash' in classes.keys() and instance['category_id'] in classes['trash']):

                    obj = {
                        "bbox": instance['bbox'],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": dict_instance_label[instance['category_id']],
                        'segmentation' : instance['segmentation']
                    }

                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)


    return dataset_dicts



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    classes = eval(args.classes_dict)
    # Register the test dataset.
    DatasetCatalog.register('test', lambda: get_dicts(args.data_dir, 'test', args.cross_val, classes))
    MetadataCatalog.get('test').set(thing_classes=list(classes.keys()))

    demo = VisualizationDemo(cfg)

    if args.input:
        '''
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        '''
        for input in tqdm.tqdm(args.input, disable=not args.output):
            stack = [None] * cfg.INPUT.STACK_SIZE
            for z in range(cfg.INPUT.STACK_SIZE):
                path = os.path.expanduser(input + cfg.INPUT.SLICE_SEPARATOR + str(z) + cfg.INPUT.EXTENSION)
                assert path, "The input path(s) was not found"
                # use PIL, to be consistent with evaluation
                stack[z] = read_image(path, format=cfg.INPUT.FORMAT)

            start_time = time.time()
            predictions, visualized_output = demo.run_on_stack(stack)

            if cfg.OUTPUT.GATHER_STACK_RESULTS:
                nb_predictions = len(predictions[0]["instances"])   # same predictions on all images/slices
            else:
                nb_predictions = 0
                for z in range(cfg.INPUT.STACK_SIZE):
                    nb_predictions += len(predictions[z]["instances"])

            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    input, nb_predictions, time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    stack_name = os.path.basename(input)
                    for z in range(cfg.INPUT.STACK_SIZE):
                        out_filename = os.path.join(args.output, stack_name + cfg.INPUT.SLICE_SEPARATOR + str(z) + cfg.INPUT.EXTENSION)
                        visualized_output[z].save(out_filename)
                else:
                    logger.info("Please specify a directory with args.output")
