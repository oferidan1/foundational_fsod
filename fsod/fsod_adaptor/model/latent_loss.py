import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(
    inputs, targets, text_mask, num_boxes = 1, alpha: float = 0.25, gamma: float = 2, no_reduction=False
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    
    if text_mask is not None:
        inputs = torch.masked_select(inputs, text_mask)
        targets = torch.masked_select(targets, text_mask)
        
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if no_reduction:
        return loss

    return loss.mean()

def token_sigmoid_binary_focal_loss(outputs, indices, alpha=0.25, gamma=2):
    pred_logits=outputs['pred_logits']
    new_targets=outputs['one_hot'].to(pred_logits.device)
    text_mask=outputs['text_mask']

    assert (new_targets.dim() == 3)
    assert (pred_logits.dim() == 3)  # batch x from x to
    
    bs, n, _ = pred_logits.shape

    if text_mask is not None:
        # ODVG: each sample has different mask 
        text_mask = text_mask.repeat(1, pred_logits.size(1)).view(outputs['text_mask'].shape[0],-1,outputs['text_mask'].shape[1])
        pred_logits = torch.masked_select(pred_logits, text_mask)
        new_targets = torch.masked_select(new_targets, text_mask)

    new_targets=new_targets.float()
    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
    p_t = p * new_targets + (1 - p) * (1 - new_targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
        loss = alpha_t * loss

    total_num_pos=0
    for batch_indices in indices:
        total_num_pos += len(batch_indices[0])
    num_pos_avg_per_gpu = max(total_num_pos , 1.0)
    loss=loss.sum()/num_pos_avg_per_gpu
            
    return loss
    
    
class ContrastiveEmbed(nn.Module):
    def __init__(self, max_text_len=256):
        """
        Args:
            max_text_len: max length of text.
        """
        super().__init__()
        self.max_text_len = max_text_len

    def forward(self, x, text_dict):
        """_summary_

        Args:
            x (_type_): _description_
            text_dict (_type_): _description_
            {
                'encoded_text': encoded_text, # bs, 195, d_model
                'text_token_mask': text_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(text_dict, dict)

        y = text_dict["encoded_text"]
        text_token_mask = text_dict["text_token_mask"]

        res = x @ y.transpose(-1, -2)
        res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        new_res = torch.full((*res.shape[:-1], self.max_text_len), float("-inf"), device=res.device)
        new_res[..., : res.shape[-1]] = res

        return new_res


