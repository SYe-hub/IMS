# -*- coding:utf-8 -*-       
import argparse
import os
import time
import random
import numpy as np

import torch
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from timm.utils.log import setup_default_logging
import pickle
from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.mixmethod import *

import warnings
import logging
from utils.ibmodel import IRM_2, IRM_3,IRM_4, IRM_5, IRM_6
warnings.filterwarnings('ignore')  
logging.getLogger().setLevel(logging.ERROR)
import torchvision.models.resnet as resnet
from scipy.spatial.distance import pdist, squareform
from utils.ib import calculate_MI
from scipy.cluster.vq import kmeans,vq,whiten
logger = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def distributed(model, device, n_gpu):
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = model.to(device)
    else:
        #assert n_gpu == 1
        model = model.to(device)
    return model


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint_CUB_%s_(domain)%s.pth" % (args.name,args.IRM_VALUE,args.domain_num))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.device = args.device

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "CDLT":
        num_classes = 258
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == 'flower':
        num_classes = 102
    elif args.dataset == 'aircraft':
        num_classes = 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,
                              smoothing_value=args.smoothing_value)
    model.load_from(np.load(args.pretrained_dir))
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
    model = distributed(model, args.device, args.n_gpu)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # cpu  vars
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        x, y,_ = batch
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


def train(args, model):
    """ Train the model """
    domain_path = f'/home/zzq/FGVC_Dataset/vit_code/vit_code/domain_CUB_{args.domain_num}.pkl'
    f_read = open(domain_path, 'rb')
    doamin_dict = pickle.load(f_read)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()
    mixmethod = args.mixmethod

    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc, best_step = 0, 0, 0
    start_time = time.time()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            x, y,img_path = batch
           
            x = x.to(args.device)
            y = y.to(args.device)
            domain_1 = []
            domain_2 = []
            domain_3 = []
            domain_4 = []
            domain_5 = []
            domain_6 = []
            

            for i in range(x.shape[0]):
                img_relativepath = os.path.join("images","/".join(img_path[i].split('/')[7:])) 
                if doamin_dict[img_relativepath] == 0:
                    domain_1.append(i)
                elif doamin_dict[img_relativepath] == 1:
                    domain_2.append(i)


            if mixmethod != 'baseline':

                input, target_a, target_b, lam_a, lam_b = eval(mixmethod)(x, y)
                logits, _ = model(input)
                loss_a = criterion(logits, target_a)
                loss_b = criterion(logits, target_b)
                loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
            else:
                loss, logits,cls_feature = model(x, y)
                loss = loss.mean().to(args.device)
            
            
                  

           
            
     
            
            # # IRM 
            if args.domain_num == 2:
                irm = IRM_2(args.IRM_VALUE)
                loss_IRM = irm(logits, y, domain_1, domain_2).to(args.device)
            elif args.domain_num == 3:
                irm = IRM_3(args.IRM_VALUE)
                loss_IRM = irm(logits, y, domain_1, domain_2,domain_3).to(args.device)
            elif args.domain_num == 4:
                irm = IRM_4(args.IRM_VALUE)
                loss_IRM = irm(logits, y, domain_1, domain_2, domain_3, domain_4).to(args.device)
            elif args.domain_num == 5:
                irm = IRM_5(args.IRM_VALUE)
                loss_IRM = irm(logits, y, domain_1, domain_2, domain_3, domain_4, domain_5).to(args.device)
            elif args.domain_num == 6:
                irm = IRM_6(args.IRM_VALUE)
                loss_IRM = irm(logits, y, domain_1, domain_2, domain_3, domain_4, domain_5,domain_6).to(args.device)
           
            
            #  MI 互信?           
            Z_numpy = cls_feature.cpu().detach().numpy()
            k = squareform(pdist(Z_numpy, 'euclidean'))  # Calculate Euclidiean distance between all samples.
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
            IXZ = calculate_MI(x, cls_feature, s_x=1000, s_y=sigma ** 2)
            
          
            Var = logits.var(dim=0).mean()
            #print("Var.shape = {}".format(Var))

            total_loss = loss + loss_IRM + 0.005 * IXZ
            
            loss = total_loss
            preds = torch.argmax(logits, dim=-1)
            

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
                all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f) (lr=%.6f)" %
                                               (global_step, t_total, losses.val, optimizer.param_groups[0]['lr']))

                if global_step % args.eval_every == 0: #and global_step > (t_total/5*2):
                # if global_step % args.eval_every == 0:
                    accuracy = valid(args, model, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                        best_step = global_step
                    logger.info("best accuracy so far: %f" % best_acc)
                    logger.info("best accuracy in step: %.0f" % best_step)
                    model.train()

                if global_step % t_total == 0:
                    break

        all_preds, all_label = all_preds[0], all_label[0]
        simple_accuracys = simple_accuracy(all_preds, all_label)
        simple_accuracys = torch.tensor(simple_accuracys).to(args.device)
        train_accuracy = simple_accuracys
        train_accuracy = train_accuracy.detach().cpu().numpy()
        logger.info("train accuracy so far: %f" % train_accuracy)
        losses.reset()
        if global_step % t_total == 0:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default='ys_fig_IRM_flower_5domain',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", "flower", "aircraft","CDLT"],
                        default="CUB_200_2011",
                        help="Which downstream task.")
    
    parser.add_argument("--data_root", type=str, default='/home/zzq/FGVC_Dataset/CUB_dataset/CUB_200_2011/')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pretrained_dir", type=str, default="/home/zzq/FGVC_Dataset/vit_code/vit_code/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="/home/zzq/FGVC_Dataset/vit_code/save_model", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=20, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default= 3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=3180*2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--mixmethod', default='baseline', type=str, help='baseline, mixup, cutout and cutmix')
    #IRM
    parser.add_argument('--IRM_VALUE',type=float,default=0.005 ,help="IRM Hyperparameter value\n")
    parser.add_argument('--domain_num',type=int,default=2,help="domain number")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup CUDA, GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    setup_default_logging(log_path=args.output_dir + '/' + args.name + '_log.log')  # so good
    logger.warning("Env GPU ids: %s, device: %s, n_gpu: %s" % (args.gpu_ids, args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Trainin    
    train(args, model)
    logger.info('\n================================ END ====================================\n')


if __name__ == "__main__":
    main()


