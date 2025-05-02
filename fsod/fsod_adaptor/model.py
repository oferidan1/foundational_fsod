import torch
import torch.nn as nn
import torch.nn.functional as F
from .latent_loss import ContrastiveEmbed


class FSOD_Adaptor(nn.Module):

    def __init__(self):

        super(FSOD_Adaptor, self).__init__()
        
        latent_size = 256
        self.MLP = nn.Sequential(nn.Linear(latent_size, latent_size), nn.GELU(), nn.Linear(latent_size, latent_size))
        self.class_embed = ContrastiveEmbed()
        self.max_text_len = 256
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, embeds, text_dict):
        embeds = self.MLP(embeds)
        outputs_class = self.class_embed(embeds, text_dict)
        
        # out = {"pred_logits": outputs_class}

        # # Used to calculate losses
        # bs, len_td = text_dict['text_token_mask'].shape
        # out['text_mask']=torch.zeros(bs, self.max_text_len, dtype=torch.bool).to(embeds.device)
        # for b in range(bs):
        #     for j in range(len_td):
        #         if text_dict['text_token_mask'][b][j] == True:
        #             out['text_mask'][b][j] = True
                    
        return outputs_class
