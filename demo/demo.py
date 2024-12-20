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
    cfg.MODEL.FCOS.NUM_CLASSES = len(eval(args.classes_dict))  #For FCOS and CondInst

    #Remove all online augmentations
    cfg.INPUT.HFLIP_TRAIN = False
    cfg.CROP.ENABLED = False
    cfg.INPUT.IS_ROTATE = False
    cfg.TEST.AUG.ENABLED = False

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
    parser.add_argument('--cross-val', default=4)

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
        img_dir = os.path.join(dir, 'Cross-val', 'Xval'+str(fold)+'_images', 'images')
        ann_dir = os.path.join(dir, 'Cross-val', 'Xval'+str(fold)+'_labels','detectron2')
    

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
    # Register the train and validation datasets.
    DatasetCatalog.register('test', lambda: get_dicts(args.data_dir, 'test', args.cross_val, classes))
    MetadataCatalog.get('test').set(thing_classes=list(classes.keys()))

    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
