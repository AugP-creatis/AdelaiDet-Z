# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

import random
import json
from tqdm import tqdm
from detectron2.structures import BoxMode


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            mapper = DatasetMapperWithBasis(cfg, True, image_format="BGR")
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # Check https://github.com/AugP-creatis/detectron2-Z/blob/main/detectron2/config/defaults.py

    # DATASETS
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("val",)
    cfg.DATASETS.TEST = ()

    # INPUT
    #cfg.INPUT.FORMAT = "BGR"
    #cfg.INPUT.MASK_FORMAT = "polygon"
    #cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    #cfg.INPUT.MAX_SIZE_TRAIN = 1333
    #cfg.INPUT.MIN_SIZE_TEST = 800
    #cfg.INPUT.MAX_SIZE_TEST = 1333

    # DATALOADER
    cfg.DATALOADER.NUM_WORKERS = 2  #As recommended

    # MODEL
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.FCOS.NUM_CLASSES = len(eval(args.classes_dict))  #For FCOS and CondInst
    #cfg.MODEL.MEInst.NUM_CLASSES = len(eval(args.classes_dict)) #For MeInst
    '''
    cfg.MODEL.PIXEL_MEAN = [218.96615195, 205.58776696, 199.45428186]
    cfg.MODEL.PIXEL_STD = [20.1397195,  19.81115748, 21.38672478]
    '''

    # SOLVER
    cfg.SOLVER.IMS_PER_BATCH = 32
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    # cfg.SOLVER.BASE_LR = 0.001
    # cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


# get_dicts and register_datasets from Nathan Hutin https://gitlab.in2p3.fr/nathan.hutin/detectron2/-/blob/main/train_cross_validation.py
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
    lenght_image_id_0 = 0
    dict_instance_label = {value:num for num, value in enumerate(classes.values())}
    list_image_non_id_0 = []
    for fold in folds_list:
        img_dir = os.path.join(dir, 'Cross-val', 'Xval'+str(fold)+'_images', 'images')
        ann_dir = os.path.join(dir, 'Cross-val', 'Xval'+str(fold)+'_labels','detectron2')
    

        for idx, file in tqdm(enumerate(os.listdir(ann_dir)), desc=f'cross validation {fold}, mode {mode}'):
            change_file_id_0 = False
            change_file_id_no_0 = False
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
            if len(dico['annotation']) == 0:
                continue
            for instance in dico['annotation']:
                if 'Trash' in classes.keys() and instance['category_id'] in classes['Trash']:
                    instance['category_id'] = 1
                if instance['category_id'] == 0 and change_file_id_0 == False:
                    lenght_image_id_0 += 1
                    change_file_id_0 = True
                if instance['category_id'] != 0 and change_file_id_no_0 == False:
                    change_file_id_no_0 = True


                if instance['category_id'] in classes.values() or ('trash' in classes.keys() and instance['category_id'] in classes['trash']):

                    obj = {
                        "bbox": instance['bbox'],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": dict_instance_label[instance['category_id']],
                        'segmentation' : instance['segmentation']
                    }

                    objs.append(obj)

            if change_file_id_0 == False and change_file_id_no_0 == True:
                list_image_non_id_0.append(record["file_name"])

            if len(objs) == 0:
                continue
            record["annotations"] = objs
            dataset_dicts.append(record)

    random.shuffle(list_image_non_id_0)
    try:
        image_remove = random.sample(list_image_non_id_0, lenght_image_id_0*2)
    except ValueError:
        image_remove =[]
    for img_rm in image_remove:
        for record in dataset_dicts:
            if record['file_name'] == img_rm:
                dataset_dicts.remove(record)
                break

    return dataset_dicts


def main(args):
    cfg = setup(args)

    classes = eval(args.classes_dict)
    # Register the train and validation datasets.
    DatasetCatalog.register('train', lambda: get_dicts(args.data_dir, 'train', args.cross_val, classes))
    DatasetCatalog.register('val', lambda: get_dicts(args.data_dir, 'val', args.cross_val, classes))
    # Set the metadata for the dataset.
    MetadataCatalog.get('train').set(thing_classes=list(classes.keys()))
    MetadataCatalog.get('val').set(thing_classes=list(classes.keys()))

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()

    parser.add_argument('--data-dir', default='/home/perrier/Bacteriocytes_seg/data')
    parser.add_argument('--classes-dict',type=str,default="{'Intact_Sharp':0, 'Broken_Sharp':2}")
    #Classes are like "{'Intact_Sharp':0,'Intact_Blurry':1,'Broken_Sharp':2,'Broken_Blurry':3}"
    parser.add_argument('--cross-val', default=4)

    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
