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


def topk_by_iou(inputs, boxes_, K=300):
    gts = []    
    gt_classes = inputs[0]['instances'].gt_classes        
    ious = compute_iou(inputs, boxes_)    
    ious = torch.from_numpy(ious)
    #loop over all bbox in gt
    topk_values, topk_idxs, topk_labels = [], [], []
    
    K_ = K // ious.shape[1] + ious.shape[1]
    #iou_thr = 0.0
    for i in range(ious.shape[1]):
        topk_idxs_ = ious[:,i].argmax().unsqueeze(0)  
        topk_values_ = ious[:,i].max().unsqueeze(0)      
        #topk_values_, topk_idxs_ = torch.topk(ious[:,i], K_, 0)
        # topk_iou_thr = topk_values_ > iou_thr
        # topk_values_ = topk_values_[topk_iou_thr]
        # topk_idxs_ = topk_idxs_[topk_iou_thr]        
        if len(topk_idxs_)>0:
            topk_values.append(topk_values_)
            topk_idxs.append(topk_idxs_)
            #topk_labels.append((gt_classes[i]*torch.ones(len(topk_idxs_))))
            topk_labels.append(gt_classes[i].unsqueeze(0) )
    
    if len(topk_values)>0:
        topk_values = torch.cat(topk_values, dim=0)
        topk_idxs = torch.cat(topk_idxs, dim=0)
        topk_labels = torch.cat(topk_labels, dim=0)
    
    return topk_values, topk_idxs, topk_labels

#rescale bb to output size
def rescale_bb(inputs):
    gt_bboxes = inputs[0]['instances'].gt_boxes    
    h_orig, w_orig = inputs[0]['height'], inputs[0]['width']
    h_real, w_real  = inputs[0]['instances'].image_size    
    h_ratio = h_orig/h_real
    w_ratio = w_orig/w_real
    bbs = []
    for b in gt_bboxes:
        bb = torch.Tensor([b[0]*w_ratio+1, b[1]*h_ratio+1, b[2]*w_ratio, b[3]*h_ratio])
        bbs.append(bb.unsqueeze(0))
    gt_bboxes_scaled = Boxes(torch.cat(bbs, dim=0))   
    return  gt_bboxes_scaled

#get GT prediction - for sanity only
def get_gt_preds(inputs):   
    gt_classes = inputs[0]['instances'].gt_classes
    gt_bboxes_scaled = rescale_bb(inputs)
    result = Instances(inputs[0]['instances'].image_size)    
    result.pred_boxes = gt_bboxes_scaled
    result.scores = torch.ones(len(gt_classes))
    result.pred_classes = gt_classes
    
    final_outputs = []
    curr_output = {}
    curr_output['instances'] = result
    final_outputs.append(curr_output)      
    
    return final_outputs, 0

SL_TP , SL_FP, SL_FN, SL_TN, SL_TOTAL = 0, 0, 0, 0, 0
def verify_pred(inputs, results, supporting_classes, iou_thr):
    global SL_TP , SL_FP, SL_FN, SL_TN, SL_TOTAL
    gt_classes = inputs[0]['instances'].gt_classes
    gt_bboxes_scaled = rescale_bb(inputs)
    
    iscrowd = [int(False)] * len(gt_bboxes_scaled)
    ious = mask_utils.iou(results.pred_boxes.tensor.tolist(), gt_bboxes_scaled.tensor.tolist(), iscrowd)
    
    for i in range(len(results.pred_classes)):
        pred_class = results.pred_classes[i]
        for j in range(len(gt_classes)):
            gt_class = gt_classes[j]
            if pred_class == supporting_classes[0]:
                if gt_class == supporting_classes[0] and ious[i][j]>=iou_thr:
                    SL_TP += 1
                else:
                    SL_FP += 1
            else:                    
                if gt_class == supporting_classes[0]:
                    SL_FN += 1
                else:
                    SL_TN += 1
        SL_TOTAL += 1
        
#save embeds
def save_embeds(inputs, embeds, scores, boxes):
    fs_create_embedding_iou = 0.5
    global embed_idx
    #novel category
    gt_classes = inputs[0]['instances'].gt_classes
    ious = compute_iou(inputs, boxes)    
    frame_info = {'embeds': embeds, 'scores': scores, "ious": ious, 'gt_class': gt_classes}            
    
    name = os.path.basename(inputs[0]["file_name"])                
    filename = 'embeds/{name}.pt'.format(name=name)
    torch.save(frame_info, filename)
    
    # # #loop over all bbox in gt
    # for i in range(ious.shape[1]):            
    #     max_iou = ious[:,i].max()     
    #     gt_class = gt_classes[i].item()      
        
    #     target_class = 7            
    #     if gt_class == target_class:
    #         with open("class7_files.txt", "a") as myfile:
    #             name = os.path.basename(inputs[0]["file_name"])
    #             to_write = "{filename}, {iou}, {gt}".format(filename=name, iou=max_iou, gt=gts)
    #             myfile.write(to_write)
    #             myfile.write("\n")            
        
    #     if max_iou > fs_create_embedding_iou and gt_class==target_class:
    #         frame_info = {'embeds': embeds, 'scores': scores, "ious": ious[:,i], 'gt_class': gt_class}            
            
    #         name = os.path.basename(inputs[0]["file_name"])                
    #         filename = 'embeds/{name}_{c}_{i}.pt'.format(name=name, c=gt_class, i=i)
    #         torch.save(frame_info, filename)
    
    
#save queries
embed_idx = {}    
def save_queries(inputs, queries, scores, boxes):
    fs_create_embedding_iou = 0.5
    gt_classes = inputs[0]['instances'].gt_classes
    ious = compute_iou(inputs, boxes)    
    #loop over all bbox in gt
    for i in range(ious.shape[1]):            
        max_iou = ious[:,i].max()     
        gt_class = gt_classes[i].item()      
        
        target_class = 7            
        if gt_class == target_class:
            with open("class7_files.txt", "a") as myfile:
                name = os.path.basename(inputs[0]["file_name"])
                to_write = "{filename}, {iou}, {gt}".format(filename=name, iou=max_iou, gt=gts)
                myfile.write(to_write)
                myfile.write("\n")            
    
        if max_iou > fs_create_embedding_iou and gt_class==target_class:
            idx = ious[:,i].argmax()                                   
            target_embed = queries[idx]                
            if gt_class not in embed_idx:
                embed_idx[gt_class] = 0            
            # name = os.path.basename(inputs[0]["file_name"])
            filename = 'queries/class{gt_class}_idx{embed_idx}_iou{max_iou:.2f}.pt'.format(gt_class=gt_class, embed_idx=embed_idx[gt_class], max_iou=max_iou)                
            # filename = 'queries/{name}.pt'.format(name=name)
            torch.save(target_embed, filename)
            embed_idx[gt_class] += 1


def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = torch.max(samples, dim=1).values
    #confidences = samples.max()
    # get predictions from confidences (positional in this case)
    predicted_label = torch.argmax(samples, dim=1)
    #predicted_label = samples.argmax()

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower &amp; upper)
        in_bin = torch.logical_and(confidences > bin_lower, confidences <= bin_upper).type(torch.int)
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.type(torch.float).mean()
        #prob_in_bin = in_bin

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].type(torch.float).mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece

def compute_iou(inputs, pred_boxes):
    h_orig, w_orig = inputs[0]['height'], inputs[0]['width']
    h_real, w_real = inputs[0]['instances'].image_size    
    h_ratio = h_orig/h_real
    w_ratio = w_orig/w_real
    gt_bboxes = inputs[0]['instances'].gt_boxes
    gt_classes = inputs[0]['instances'].gt_classes     
    gt_bb = []  
    for b in gt_bboxes:
        gt_bb.append([b[0]*w_ratio+1, b[1]*h_ratio+1, b[2]*w_ratio, b[3]*h_ratio])          
    iscrowd = [int(False)] * len(gt_bboxes)
    ious = mask_utils.iou(pred_boxes.tensor.tolist(), gt_bb, iscrowd)
    return ious


def compute_calibration_error(inputs, pred_box_probs, pred_box_coords, iou_threshold=0.5):
    # gt_classes: (num_gt_boxes) gives the class id for each ground truth box
    # gt_coords: (num_gt_boxes, 4) gives the coordinates for each ground truth box
    # pred_box_probs: (num_pred_boxes, num_classes) gives the logits for each predicted box
    # pred_coords: (num_pred_boxes, 4) gives the coordinates for each predicted box

    # Step 1: compute the  ground truth label for each predicted box
    ious = compute_iou(inputs, pred_box_coords)
    gt_classes = inputs[0]['instances'].gt_classes     
    matching_gt_box_ids = np.argmax(ious, axis=1) # (num_gt_boxes) - gives the pred box id for each gt box        
    gt_box_classes = torch.ones(pred_box_coords.tensor.shape[0], gt_classes.shape[0])*gt_classes
    pred_box_gt_labels = gt_box_classes[0][matching_gt_box_ids] # assign each pred box the class of the gt box with the highest iou
    pred_box_gt_labels[ious.max(axis=1) < iou_threshold] = -1  # set to -1 if no gt box has iou > 0.5
    # Step 2: compute the expected calibration error (ECE) for each binary classifier
    # Note: later this can be extended to other types of calibration error
    num_classes = pred_box_probs.shape[1]
    eces = np.zeros(num_classes)
    
    for i in range(num_classes):
        # get the ground truth labels for class i, for each box by setting all other classes to 0
        labels_i = (pred_box_gt_labels == i).type(torch.float32) # (num_pred_boxes) 
        # get the predicted probabilities for class i
        probs_i = pred_box_probs[:, i].unsqueeze(0) # (num_pred_boxes) - the sigmoids for category i
        # transform into binary-like probabilities for ECE function
        probs_i = torch.cat([1-probs_i, probs_i], dim=0).T # (num_pred_boxes, 2)
        # compute the ECE
        eces[i] = expected_calibration_error(probs_i, labels_i)

    return eces    

def compute_calibration_error_mine(inputs, pred_box_probs, pred_box_coords, iou_threshold=0.5):
    # gt_classes: (num_gt_boxes) gives the class id for each ground truth box
    # gt_coords: (num_gt_boxes, 4) gives the coordinates for each ground truth box
    # pred_box_probs: (num_pred_boxes, num_classes) gives the logits for each predicted box
    # pred_coords: (num_pred_boxes, 4) gives the coordinates for each predicted box

    # Step 1: compute the  ground truth label for each predicted box
    ious = compute_iou(inputs, pred_box_coords)
    gt_classes = inputs[0]['instances'].gt_classes     
    matching_gt_box_ids = np.argmax(ious, axis=0) # (num_gt_boxes) - gives the pred box id for each gt box    
    pred_box_labels = torch.ones(pred_box_coords.tensor.shape[0], gt_classes.shape[0])
    for i in range(len(gt_classes)):
        pred_box_labels[:,i] *=  gt_classes[i]
        #pred_box_labels[matching_gt_box_ids[i],i] = gt_classes[i]
    #gt_box_classes[matching_gt_box_ids] = 1
    #pred_box_labels = gt_box_classes[matching_gt_box_ids] # assign each pred box the class of the gt box with the highest iou
    pred_box_labels[ious.max(axis=1) < iou_threshold] = -1  # set to -1 if no gt box has iou > 0.5

    # Step 2: compute the expected calibration error (ECE) for each binary classifier
    # Note: later this can be extended to other types of calibration error
    num_classes = pred_box_probs.shape[1]
    eces = np.zeros(num_classes)
    
    j = 0
    for i in range(num_classes):
        # get the ground truth labels for class i, for each box by setting all other classes to 0
        if j<len(gt_classes) and gt_classes[j] == i:        
            gt_labels_i = (pred_box_labels[:,j] == i).type(torch.float32) # (num_pred_boxes) 
            j += 1
        else:
            gt_labels_i = torch.zeros(pred_box_labels.shape[0])            
        # get the predicted probabilities for class i
        pred_probs_i = pred_box_probs[:, i].unsqueeze(0)# (num_pred_boxes)
        # transform into binary-like probabilities for ECE function
        pred_probs_i = torch.cat((1-pred_probs_i, pred_probs_i), axis=0).T # (num_pred_boxes, 2)
        # compute the ECE
        eces[i] = expected_calibration_error(pred_probs_i, gt_labels_i)

    return eces

        
# run gdino model with params
def run_gdino(model, inputs, text_prompt_list, positive_map_list, is_create_fs, is_gt_iou, is_ece, K, score_thr, dataset_classes, iou_thr=0.7):
    #return get_gt_preds(inputs)
    length = 81
    image, image_src = prepare_image_for_GDINO(inputs[0])
    start_compute_time = time.time()
    filename = os.path.basename(inputs[0]["file_name"])
    
    if len(text_prompt_list)>1: #LVIS data has 1203 classes
        image1 = image.repeat(8, 1, 1, 1)        
        image2 = image.repeat(7, 1, 1, 1)            
        with torch.no_grad():    
            output1 = model(image1, captions = text_prompt_list[0:8])
            output2 = model(image2, captions = text_prompt_list[8:])
        
        if len(output1['original_matched_boxes']):
            outputs = {'pred_logits':torch.cat((output1['pred_logits'], output2['pred_logits'])), 
                        'pred_boxes': torch.cat((output1['pred_boxes'], output2['pred_boxes'])), 
                        'pred_embeds': torch.cat((output1['pred_embeds'], output2['pred_embeds'])),
                        'pred_queries': torch.cat((output1['pred_queries'], output2['pred_queries'])),
                        'original_matched_boxes': torch.cat((output1['original_matched_boxes'], output2['original_matched_boxes'])),
                        'original_matched_boxes_classes': torch.cat((output1['original_matched_boxes_classes'], output2['original_matched_boxes_classes']))}    
        else:
            outputs = {'pred_logits':torch.cat((output1['pred_logits'], output2['pred_logits'])), 
                  'pred_boxes': torch.cat((output1['pred_boxes'], output2['pred_boxes'])), 
                  'pred_embeds': torch.cat((output1['pred_embeds'], output2['pred_embeds'])),
                  'pred_queries': torch.cat((output1['pred_queries'], output2['pred_queries'])),
                  'original_matched_boxes': [], 'original_matched_boxes_classes': []}
        
    else:
        with torch.no_grad():
            outputs = model(image, captions=text_prompt_list, iou_thr=iou_thr, filename=filename)
        
    torch.cuda.synchronize()
    total_compute_time = time.time() - start_compute_time
    out_logits = outputs["pred_logits"]  # prediction_logits.shape = (batch, nq, 256)
    out_bbox = outputs["pred_boxes"] # prediction_boxes.shape = (batch, nq, 4)    
    out_embeds = outputs["pred_embeds"]
    out_queries = outputs["pred_queries"]
    matched_boxes_idx = outputs["original_matched_boxes"]
    matched_boxes_classes = outputs["original_matched_boxes_classes"]    
    prob_to_token = out_logits.sigmoid() # prob_to_token.shape = (batch, nq, 256)
    boxes = []
    
    thr_gain = 2
    cat_batch_id = 0 #= fs_category_id // positive_map_list[0].shape[0]    
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
    prob_to_label = torch.cat(prob_to_label_list, dim = 1) # shape: (nq, number of categories)                  
    
    if is_gt_iou:    
        topk_values, topk_idxs, topk_labels = topk_by_iou(inputs, out_bbox.squeeze(0), K)    
    else:
        topk_values, topk_idxs = torch.topk(prob_to_label.view(-1), K, 0)      
        
    h, w = inputs[0]['height'], inputs[0]['width']    
    #take boxes with minimal similarity score
    topk_values_thr = topk_values>score_thr
    if sum(topk_values_thr)>0:
        topk_values = topk_values[topk_values_thr]
        topk_idxs = topk_idxs[topk_values_thr]        
    
        # # topk_idxs contains the index of the flattened tensor. We need to convert it to the index in the original tensor
        scores = topk_values # Shape: (300,)
        topk_boxes = topk_idxs // prob_to_label.shape[1] # to determine the index in 'num_query' dimension. Shape: (300,)
        labels = topk_idxs % prob_to_label.shape[1] # to determine the index in 'num_category' dimension. Shape: (300,)
        topk_boxes_batch_idx = labels // length # to determine the index in 'batch_size' dimension. Shape: (300,)
        combined_box_index = torch.stack((topk_boxes_batch_idx, topk_boxes), dim=1)    
        if len(out_bbox.shape) < 3:
            out_bbox = out_bbox.unsqueeze(0)
        boxes = out_bbox[combined_box_index[:, 0], combined_box_index[:, 1]].to("cpu") # Shape: (300, 4)
        embeds = out_embeds[combined_box_index[:, 0], combined_box_index[:, 1]].to("cpu") # Shape: (300, 4)    
        if is_gt_iou:    
            boxes = out_bbox.squeeze(0)[topk_idxs].to("cpu")                         
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes = boxes, in_fmt = "cxcywh", out_fmt = "xyxy")
        if is_gt_iou:    
            labels = topk_labels
        else:
            labels = labels.to(torch.int64)
            # topk_values, topk_idxs = torch.topk(scores, K)
            # labels = labels[topk_idxs]    
        scores = topk_values    
        boxes = Boxes(boxes)
        #debug: diningtable=7
        #labels+=7
    else:
        boxes, scores, labels = [], [], []
    
    eces = 0
    if is_ece:
        eces = compute_calibration_error(inputs, prob_to_label, boxes)
        
    result = Instances((h, w))
    result.pred_boxes = boxes
    result.scores = scores
    result.pred_classes = labels
    
    final_outputs = []
    curr_output = {}
    curr_output['instances'] = result
    final_outputs.append(curr_output)          
    
    # if len(model.supporting_classes):
    #     verify_pred(inputs, result, model.supporting_classes, 0.5)
    
    #visualization
    #do_gdino_visualization(model, text_prompt_list, image_src, inputs[0]["file_name"], scores, boxes, labels, dataset_classes)
    
    # saving embedding of queries to file
    # will be used later as few shots
    if is_create_fs:
        save_embeds(inputs, embeds, scores, boxes)
        
    return final_outputs, total_compute_time, eces

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
    eces_list = []
    with open("class7_files_test_names.txt") as f:
        class7_names = f.read()
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            
            #run only on classs7 images
            filename = os.path.basename(inputs[0]['file_name'])
            if args.is_class7 and filename not in class7_names:
                continue            
            
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
                outputs, compute_time, eces = run_gdino(model, inputs, text_prompt_list, positive_map_list, args.is_create_fs, args.is_gt_iou, args.is_ece, args.topk, args.score_thr, dataset_classes)
                total_compute_time += compute_time
                eces_list.append(eces)
                
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


    eces_mean = np.mean(eces_list, axis=0)
    print("-----------------------------")
    print("mean ECEs: {}".format(eces_mean))
    print("-----------------------------")
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
    
    # global SL_TP , SL_FP, SL_FN, SL_TN, SL_TOTAL
    # print("TP: {}, FP: {}, FN: {}".format(SL_TP , SL_FP, SL_FN))
    # print("Accuracy: {}", SL_TP/(SL_FP+SL_FN))

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
