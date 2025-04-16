import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch
from detectron2.utils.comm import is_main_process
from detectron2.structures import Instances, Boxes, pairwise_iou
from torchvision.ops import box_convert
import transforms as T2
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw, ImageFont
import os
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(
                        k
                    )
                    results[k] = v
        return results


def inference_on_dataset_orig(model, data_loader, evaluator):
#def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time            
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(
                        seconds_per_img * (total - num_warmup) - duration
                    )
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(
        datetime.timedelta(seconds=int(total_compute_time))
    )
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def prepare_image_for_GDINO(input, device = "cuda"):
    """
    inputs: dict, with keys "file_name", "height", "width", "image", "image_id"
    outputs: transformed images
    """
    transform = T2.Compose(
        [
            T2.RandomResize([800], max_size=1333),
            T2.ToTensor(),
            T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    image_src = Image.open(input["file_name"]).convert("RGB")
    image = np.asarray(image_src)
    image_transformed, _ = transform(image_src, None)
    image_transformed = image_transformed.to(device)
    return image_transformed[None], image_src

# fs_gdino

#visualiztion
def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        #box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        #box[:2] -= box[2:] / 2
        #box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def do_gdino_visualization(model, caption, image_pil, filename, scores, boxes, labels, dataset_classes):
    text_threshold = 0.15
    box_threshold = 0.2
    logits_filt = scores.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    labels_filt = labels.cpu().clone()
    filt_mask = scores > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    labels_filt = labels_filt[filt_mask]
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box, label in zip(logits_filt, boxes_filt, labels_filt):
        pred_phrases.append(dataset_classes[label] + f"({str(logit.max().item())[:4]})")
            
     # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }    
        
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    
    name = os.path.basename(filename)
    output_dir = 'visualization'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_with_box.save(os.path.join(output_dir, name))

# run gdino model with params
embed_idx = {}
def run_gdino(model, inputs, text_prompt_list, positive_map_list, is_create_fs, dataset_classes, iou_thr=0.7):
    K = 100
    #K = 900
    length = 81
    image, image_src = prepare_image_for_GDINO(inputs[0])
    start_compute_time = time.time()
    filename = os.path.basename(inputs[0]["file_name"])
    with torch.no_grad():
        outputs = model(image, captions=text_prompt_list, iou_thr=iou_thr, filename=filename)
    torch.cuda.synchronize()
    total_compute_time = time.time() - start_compute_time
    out_logits = outputs["pred_logits"]  # prediction_logits.shape = (batch, nq, 256)
    out_bbox = outputs["pred_boxes"] # prediction_boxes.shape = (batch, nq, 4)    
    out_embeds = outputs["pred_queries"]
    matched_boxes_idx = outputs["original_matched_boxes"]
    matched_boxes_classes = outputs["original_matched_boxes_classes"]    
    prob_to_token = out_logits.sigmoid() # prob_to_token.shape = (batch, nq, 256)
    
    thr_gain = 10
    cat_batch_id = 0 #= fs_category_id // positive_map_list[0].shape[0]
    # fs_category_id = matched_boxes_classes
    # cat_offset_id = fs_category_id % positive_map_list[0].shape[0]
    # cat_indexes = torch.where(positive_map_list[cat_batch_id][cat_offset_id]>0)[0]
    # cols = torch.zeros(prob_to_token[0].shape).to(prob_to_token.device)    
    # cols[:,cat_indexes] = prob_to_token[cat_batch_id][:,cat_indexes]
    # rows = torch.zeros(prob_to_token[0].shape).to(prob_to_token.device)#*(cat_sim.squeeze(1) > thr_cat).unsqueeze(1)        
    # rows[matched_boxes_idx,:] = prob_to_token[cat_batch_id][matched_boxes_idx, :]
    # prob_to_token[cat_batch_id] = prob_to_token[cat_batch_id] + rows*cols*thr_gain   
    
    
    #loop over matched_boxes_idx and update relevant class prob
    for i in range(len(matched_boxes_idx)):
        box_id = matched_boxes_idx[i]
        class_id = matched_boxes_classes[i]
        cat_offset_id = class_id % positive_map_list[0].shape[0]
        cat_id = torch.where(positive_map_list[cat_batch_id][cat_offset_id]>0)[0]
        prob_to_token[cat_batch_id,box_id, cat_id] = prob_to_token[cat_batch_id,box_id, cat_id] * thr_gain             
    
    prob_to_label_list = []    
    for i in range(prob_to_token.shape[0]):
        # (nq, 256) @ (num_categories, 256).T -> (nq, num_categories)
        curr_prob_to_label = prob_to_token[i] @ positive_map_list[i].to(prob_to_token.device).T
        prob_to_label_list.append(curr_prob_to_label.to("cpu"))    
    prob_to_label = torch.cat(prob_to_label_list, dim = 1) # shape: (nq, 1203)                  
    topk_values, topk_idxs = torch.topk(
        prob_to_label.view(-1), K, 0
    )
    #topk_idxs contains the index of the flattened tensor. We need to convert it to the index in the original tensor
    scores = topk_values # Shape: (300,)
    topk_boxes = topk_idxs // prob_to_label.shape[1] # to determine the index in 'num_query' dimension. Shape: (300,)
    labels = topk_idxs % prob_to_label.shape[1] # to determine the index in 'num_category' dimension. Shape: (300,)
    topk_boxes_batch_idx = labels // length # to determine the index in 'batch_size' dimension. Shape: (300,)
    combined_box_index = torch.stack((topk_boxes_batch_idx, topk_boxes), dim=1)
    boxes = out_bbox[combined_box_index[:, 0], combined_box_index[:, 1]].to("cpu") # Shape: (300, 4)
    embeds = out_embeds[combined_box_index[:, 0], combined_box_index[:, 1]]
    h, w = inputs[0]['height'], inputs[0]['width']
    boxes = boxes * torch.Tensor([w, h, w, h])
    boxes = box_convert(boxes = boxes, in_fmt = "cxcywh", out_fmt = "xyxy")

    labels = labels.to(torch.int64)

    topk_scores, topk_idxs = torch.topk(scores, K)
    labels = labels[topk_idxs]
    scores = topk_scores
    
    result = Instances((h, w))
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = labels
    
    final_outputs = []
    curr_output = {}
    curr_output['instances'] = result
    final_outputs.append(curr_output)           
    
    #visualization
    #do_gdino_visualization(model, text_prompt_list, image_src, inputs[0]["file_name"], scores, boxes, labels, dataset_classes)
    
    # saving embedding of queries to file
    # will be used later as few shots
    fs_create_embedding_iou = 0.5
    if is_create_fs:    
        global embed_idx
        #novel category
        gts = []
        gt_bboxes = inputs[0]['instances'].gt_boxes
        gt_classes = inputs[0]['instances'].gt_classes
        for b in gt_bboxes:
            gts.append([b[0], b[1], b[0]+b[2], b[1]+b[3]])        
        iscrowd = [int(False)] * len(gt_bboxes)
        ious = mask_utils.iou(boxes.tolist(), gts, iscrowd)
        #loop over all bbox in gt
        for i in range(ious.shape[1]):
            max_iou = ious[:,i].max()     
            gt_class = gt_classes[i].item()      
            
            target_class = 2            
            if gt_class == target_class:
                with open("class2_files.txt", "a") as myfile:
                    name = os.path.basename(inputs[0]["file_name"])
                    to_write = "{filename}, {iou}, {gt}".format(filename=name, iou=max_iou, gt=gts)
                    myfile.write(to_write)
                    myfile.write("\n")
            
            if max_iou > fs_create_embedding_iou and gt_class==target_class:
                idx = ious[:,i].argmax()   
                target_embed = embeds[idx]                
                if gt_class not in embed_idx:
                    embed_idx[gt_class] = 0            
                #filename = 'queries/class{gt_class}_idx{embed_idx}_iou{max_iou:.2f}.pt'.format(gt_class=gt_class, embed_idx=embed_idx[gt_class], max_iou=max_iou)
                name = os.path.basename(inputs[0]["file_name"])
                filename = 'queries/{name}.pt'.format(name=name)
                torch.save(target_embed, filename)
                embed_idx[gt_class] += 1
        
    return final_outputs, total_compute_time

def inference_on_dataset(model, data_loader, text_prompt_list, positive_map_list, evaluator, dataset_classes, args):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50    
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0
            
            # if open set prompt - call open detector, gdino              
            if text_prompt_list == None:                     
                start_compute_time = time.time()       
                outputs = model(inputs)
                torch.cuda.synchronize()
                total_compute_time += time.time() - start_compute_time            
            else:
                # fs_gdino
                outputs, compute_time = run_gdino(model, inputs, text_prompt_list, positive_map_list, args.is_create_fs, dataset_classes)
                total_compute_time += compute_time
                
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(
                        seconds_per_img * (total - num_warmup) - duration
                    )
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(
        datetime.timedelta(seconds=int(total_compute_time))
    )
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
