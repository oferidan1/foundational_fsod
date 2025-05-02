"""
Entry point training and testing TransPoseNet
"""
import argparse
import torch
import numpy as np
from model.model import FSOD_Adaptor
from latent_dataset import LatentDataset
from model.latent_loss import sigmoid_focal_loss
import os

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", help="train or eval", default='train')
    arg_parser.add_argument("--dataset_path", help="path to the location of the dataset", default="/mnt/d/ofer/vlm/foundational_fsod/fsod/embeds/")
    arg_parser.add_argument("--lr", type=float, help="lr", default="1e-4")
    arg_parser.add_argument("--eps", type=float, help="eps", default="1e-10")
    arg_parser.add_argument("--weight_decay", type=float, help="weight_decay", default="1e-4")
    arg_parser.add_argument("--lr_scheduler_step_size", type=float, help="lr_scheduler_step_size", default="0.1")
    arg_parser.add_argument("--lr_scheduler_gamma", type=float, help="lr_scheduler_gamma", default="10")
    arg_parser.add_argument("--batch_size", type=int, help="batch_size", default="128")
    arg_parser.add_argument("--n_workers", type=int, help="n_workers", default="4")
    arg_parser.add_argument("--epochs", type=int, help="epochs", default="20")    
    arg_parser.add_argument("--n_freq_print", type=int, help="n_freq_print", default="10")        
    arg_parser.add_argument("--gpu", help="gpu id", default="0")
    arg_parser.add_argument("--output", help="output dir", default="output")

    args = arg_parser.parse_args()

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = 'cuda:' + args.gpu
    np.random.seed(numpy_seed)
    device = torch.device(device_id)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Create the model
    model = FSOD_Adaptor().to(device)

    if args.mode == 'train':
        # Set to train mode
        model.train()
        # Set the optimizer and scheduler
        params = list(model.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_scheduler_step_size, args.lr_scheduler_gamma)
        
        dataset = LatentDataset(args.dataset_path)
        loader_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.n_workers}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        
        text_dict = torch.load(os.path.join(args.dataset_path, 'text_embed_7.pth'), weights_only=True)
        
        # Used to calculate losses
        bs, len_td = text_dict['text_token_mask'].shape
        text_mask=torch.zeros(bs, model.max_text_len, dtype=torch.bool).to(device)
        for b in range(bs):
            for j in range(len_td):
                if text_dict['text_token_mask'][b][j] == True:
                    text_mask[b][j] = True
        
        for epoch in range(args.epochs):        
            for batch_idx, (EP, HP, EN, HN) in enumerate(dataloader):
                EP = EP.to(device)                
                HP = HP.to(device)                
                EN = EN.to(device)                
                HN = HN.to(device)                

                # Zero the gradients
                optim.zero_grad()

                # Forward model
                outputs_EP = model(EP, text_dict).squeeze(0)
                outputs_HP = model(HP, text_dict).squeeze(0)
                outputs_EN = model(EN, text_dict).squeeze(0)
                outputs_HN = model(HN, text_dict).squeeze(0)     
                
                # calc loss
                loss_EP = sigmoid_focal_loss(inputs=outputs_EP, targets=torch.ones_like(outputs_EP), text_mask=text_mask)
                loss_HP = sigmoid_focal_loss(inputs=outputs_HP, targets=torch.ones_like(outputs_HP), text_mask=text_mask)
                loss_EN = sigmoid_focal_loss(inputs=outputs_EN, targets=torch.zeros_like(outputs_EN), text_mask=text_mask)
                loss_HN = sigmoid_focal_loss(inputs=outputs_HN, targets=torch.zeros_like(outputs_HN), text_mask=text_mask)
                
                loss = loss_EP + loss_HP + loss_EN + loss_HN
                
                #backprop
                loss.backward()
                optim.step()
                
                if batch_idx % args.n_freq_print == 0:                
                    print("loss: "+ str(loss.item()))
                
            # Scheduler update
            scheduler.step()
            
            
        #save checkpoint
        checkpoint_path = os.path.join(args.output, "fsod_adaptor.pt")
        torch.save(model.state_dict(),checkpoint_path)

