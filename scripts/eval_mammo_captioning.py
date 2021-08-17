import sys
sys.path.append(".")

import argparse
import json
import os
from typing import Any, Dict, List

from loguru import logger
import torch
from torch.utils.data import DataLoader

from virtex.config import Config
from virtex.data import MammoDirectoryDataset, MammoCaptioningDataset, KVGHCaptioningDataset
from virtex.factories import TokenizerFactory, PretrainingModelFactory, ImageTransformsFactory
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser
from virtex.utils.metrics import CocoCaptionsEvaluator

import albumentations as alb
import cv2


# fmt: off
parser = common_parser(
    description="""Run image captioning inference on a pretrained model, and/or
    evaluate pretrained model on COCO Captions val2017 split."""
)
parser.add_argument(
    "--data-root", default=None,
    help="""Path to a directory containing image files to generate captions for.
    Default: COCO val2017 image directory as expected relative to project root."""
)
parser.add_argument(
    "--csv-path", default=None,
    help="""Path to a directory containing image files to generate captions for.
    Default: COCO val2017 image directory as expected relative to project root."""
)
parser.add_argument(
    "--checkpoint-path", required=True,
    help="Path to load checkpoint and run captioning evaluation."
)
parser.add_argument(
    "--output", default=None,
    help="Path to save predictions as a JSON file."
)
parser.add_argument(
    "--calc-metrics", action="store_true",
    help="""Calculate CIDEr and SPICE metrics using ground truth COCO Captions.
    This flag should not be set when running inference on arbitrary images."""
)
# fmt: on


def main(_A: argparse.Namespace):
    print(_A)
    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device (this will be zero here by default).
        device = torch.cuda.current_device()

    _C = Config(_A.config, _A.config_override)

    tokenizer = TokenizerFactory.from_config(_C)

    if _A.data_root is None:
        _A.data_root = os.path.join(_C.DATA.ROOT, "val2017")

    #transforms = ImageTransformsFactory.from_config(_C)
    image_transform_list: List[Callable] = []

    for name in getattr(_C.DATA, f"IMAGE_TRANSFORM_VAL"):
        # Pass dimensions if cropping / resizing, else rely on the defaults
        # as per `ImageTransformsFactory`.
        print(_C.MODEL.NAME, (_C.MODEL.NAME != "mammo"))
        if ("resize" in name or "crop" in name) and (_C.MODEL.NAME != "mammo"):
            image_transform_list.append(
                ImageTransformsFactory.create(name, _C.DATA.IMAGE_CROP_SIZE)
            )
        else:
            image_transform_list.append(ImageTransformsFactory.create(name))

    transforms = alb.Compose(image_transform_list)
    if _A.csv_path is None:
        val_dataset = MammoDirectoryDataset(_A.data_root, image_transform=transforms)
    else:
        val_dataset = KVGHCaptioningDataset(_A.data_root, _A.csv_path, tokenizer=tokenizer, image_transform=transforms, max_caption_length=_C.DATA.MAX_CAPTION_LENGTH)
    '''val_dataloader = DataLoader(
        ImageDirectoryDataset(_A.data_root),
        batch_size=_C.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )'''
    # Initialize model from a checkpoint.
    model = PretrainingModelFactory.from_config(_C)
    model.visual.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.to(device)
    ITERATION = CheckpointManager(model=model).load(_A.checkpoint_path)
    model.eval()

    # Make a list of predictions to evaluate.
    predictions: List[Dict[str, Any]] = []

    #for val_iteration, val_entry in enumerate(val_dataset, start=1):
    for idx in range(len(val_dataset)):
        val_entry = {}
        val_entry["image_id"] = val_dataset[idx]["image_id"].to(device)
        val_entry["l_cc_img"] = val_dataset[idx]["l_cc_img"].to(device)
        val_entry["l_mlo_img"] = val_dataset[idx]["l_mlo_img"].to(device)
        val_entry["r_cc_img"] = val_dataset[idx]["r_cc_img"].to(device)
        val_entry["r_mlo_img"] = val_dataset[idx]["r_mlo_img"].to(device)
        val_batch = val_dataset.collate_fn([val_entry])
        #val_batch["l_cc_img"] = val_batch["l_cc_img"].to(device)
        #val_batch["l_mlo_img"] = val_batch["l_mlo_img"].to(device)
        #val_batch["r_cc_img"] = val_batch["r_cc_img"].to(device)
        #val_batch["r_mlo_img"] = val_batch["r_mlo_img"].to(device)

        with torch.no_grad():
            output_dict = model(val_batch)

        # Make a dictionary of predictions in COCO format.
        for image_id, caption, l_cc_img, l_mlo_img, r_cc_img, r_mlo_img in zip(
            val_batch["image_id"], output_dict["predictions"], val_batch["l_cc_img"], val_batch["l_mlo_img"], val_batch["r_cc_img"], val_batch["r_mlo_img"]
        ):
            predictions.append(
                {
                    # Convert image id to int if possible (mainly for COCO eval).
                    "image_id": image_id.item(),
                    "image_path": os.path.split(val_dataset[idx]['l_cc_path'])[0],
                    "caption": tokenizer.decode(caption.tolist()),
                }
            )
            #print(predictions[-1]["image_id"], predictions[-1]["caption"])
            vis = ((torch.cat((torch.cat((r_cc_img, l_cc_img), dim=2), torch.cat((r_mlo_img, l_mlo_img), dim=2)), dim=1) * val_dataset.std[0] + val_dataset.mean[0])*255).cpu().numpy()[0]
            #cap = ' '.join(word for word in predictions[-1]["caption"])
            #print(predictions[-1]["image_id"], predictions[-1]["caption"], len(predictions[-1]["caption"]), cap)
            cv2.putText(vis, predictions[-1]["image_path"], (10, vis.shape[0]-70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
            cv2.putText(vis, 'Annotation - '+val_dataset[idx]["caption"], (10, vis.shape[0]-45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
            cv2.putText(vis, 'Prediction - '+predictions[-1]["caption"], (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join('testout', str(predictions[-1]["image_id"])+'.jpg'), vis)

    logger.info("Displaying first 50 caption predictions:")
    for pred in predictions[:50]:
        logger.info(f"{pred['image_id']} :: {pred['caption']}")

    # Save predictions as a JSON file if specified.
    if _A.output is not None:
        os.makedirs(os.path.dirname(_A.output), exist_ok=True)
        json.dump(predictions, open(_A.output, "w"))
        logger.info(f"Saved predictions to {_A.output}")

    # Calculate CIDEr and SPICE metrics using ground truth COCO Captions. This
    # should be skipped when running inference on arbitrary images.
    if _A.calc_metrics:
        # Assume ground truth (COCO val2017 annotations) exist.
        gt = os.path.join(_C.DATA.ROOT, "annotations", "captions_val2017.json")

        metrics = CocoCaptionsEvaluator(gt).evaluate(predictions)
        logger.info(f"Iter: {ITERATION} | Metrics: {metrics}")


if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.num_gpus_per_machine > 1:
        raise ValueError("Using multiple GPUs is not supported for this script.")

    # No distributed training here, just a single process.
    main(_A)
