import argparse
import glob
import logging
import math
import os
import platform
import random
import time
import json
from pathlib import Path
from threading import Thread

from torchvision import datasets
import torchvision.transforms as transforms

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as tdata
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
#import lpips

import cv2
from tqdm import tqdm, trange
import numpy as np
from utils.model import *
from utils.datasets import *
from utils.loss import *
from utils.general import *
from utils.torch_utils import *
from utils.metrics import *
from utils.utils import *

logger = logging.getLogger(__name__)

def update_stats(dict, writer, trainstats, valstats, epoch):
    dict["trainloss"].append(trainstats[0])
    dict["trainpsnr"].append(trainstats[1])
    dict["trainssim"].append(trainstats[2])
    writer.add_scalar('Loss/train',trainstats[0],global_step=epoch)
    writer.add_scalar('PSNR/train',trainstats[1],global_step=epoch)
    writer.add_scalar('SSIM/train',trainstats[2],global_step=epoch)
    dict["valloss"].append(valstats[0])
    dict["valpsnr"].append(valstats[1])
    dict["valssim"].append(valstats[3])
    if math.prod(valstats)>0:
        writer.add_scalar('Loss/val',valstats[0],global_step=epoch)
        writer.add_scalar('PSNR/val',valstats[1],global_step=epoch)
        writer.add_scalar('SSIM/val',valstats[3],global_step=epoch)

def process_loss(Y_p, Y_pred, Y, loss_fn, loss_fn_perc, loss_fn_dft, loss_fn_ssim):
    if type(Y_p) is tuple:
        loss  = loss_fn(Y_p[1],Y)         # calculate primary loss
        loss0 = loss_fn(Y_p[0],Y)
        loss += loss0
    else:
        loss  = loss_fn(Y_p,Y)

    if loss_fn_perc is not None:
        loss_p = loss_fn_perc(Y_pred,Y)   # calculate perceptual loss
        loss += loss_p
    if loss_fn_dft is not None:
        loss_d = loss_fn_dft(Y_pred,Y)    # calculate dft loss
        loss += loss_d
    if loss_fn_ssim is not None:
        loss_s = loss_fn_ssim(Y_pred,Y)   # calculate ssim loss
        loss += loss_s

    return loss

def train_step(model, device, epochs_, epoch, dataloader, optimizer, loss_fn, loss_fn_perc=None, loss_fn_dft=None, loss_fn_ssim=None):

    # Select mode
    model.train()

    # Initialize metrices
    trainloss, trainlosslq, trainpsnr, trainpsnrlq, trainssim = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, (X, Y) in enumerate(dataloader):
        
        volume, bsize = math.prod(Y.shape), Y.shape[0]
        # load data to GPU/CPU
        X, Y = X.to(device), Y.to(device)
        
        Y_p = model(X)                        # forward
        Y_pred = Y_p[1] if type(Y_p) is tuple else Y_p
        
        # update
        loss = process_loss(Y_p, Y_pred, Y, loss_fn, loss_fn_perc, loss_fn_dft, loss_fn_ssim)

        X = upscale(X, Y)
        loss_hq = nn.MSELoss(reduction='sum')(Y_pred,Y)
        loss_lq = nn.MSELoss(reduction='sum')(X,Y)
        trainloss += loss_hq.item()           # accumulate L2 loss
        trainlosslq += loss_lq.item()

        optimizer.zero_grad()                 # init backprop
        loss.backward()                       # backward
        
        # nn.utils.clip_grad_norm_(model.parameters(), 0.4) 
        optimizer.step()                      # optimize pro iteration

        mem = torch.cuda.memory_reserved()/(1024**3) if torch.cuda.is_available() else 0

        # calculate psnr and ssim
        psnr_lq = 10*math.log10(2*volume/(trainlosslq/(i+1)))
        trainpsnrlq += psnr_lq
        psnr_hq = 10*math.log10(2*volume/(trainloss/(i+1)))
        trainpsnr += psnr_hq
        trainssim += ssim_(Y_pred,Y)

        # Progress bar
        epochs_.set_description(f"Epoch: {epoch+1}, Itr: {i+1}, GPU: {mem} GB ", refresh=True)
        epochs_.set_postfix(Loss=trainloss/(i+1), AvgPSNRLow=psnr_lq, AvgPSNRHigh=psnr_hq)

    # Adjust metrics to get average loss and snr per batch 
    trainloss = trainloss/len(dataloader)
    trainpsnr = trainpsnr/len(dataloader)
    trainssim = trainssim/len(dataloader)
    return trainloss, trainpsnr, trainssim

def val_step(model, device, dataloader, loss_fn, writer):

    # Select mode
    model.eval() 
    
    # Initialize metrices
    valloss, vallosslq, valpsnr, valpsnrlq, valssim = 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for i, (X, Y) in enumerate(dataloader):

            volume, bsize = math.prod(Y.shape), Y.shape[0]
            # send data to target device
            X, Y = X.to(device), Y.to(device)
    
            # Forward pass
            Y_p = model(X)
            Y_pred = Y_p[1] if type(Y_p) is tuple else Y_p

            # calculate and accumulate loss
            X = upscale(X, Y)            
            loss_hq = nn.MSELoss(reduction='sum')(Y_pred,Y)
            loss_lq = nn.MSELoss(reduction='sum')(X,Y)
            valloss += loss_hq.item()
            vallosslq += loss_lq.item()

            mem = torch.cuda.memory_reserved()/(1024**3) if torch.cuda.is_available() else 0
            
            # calculate psnr and ssim           
            valpsnrlq += 10*math.log10(2*volume/(vallosslq/(i+1)))     # psnr(X,Y)
            valpsnr += 10*math.log10(2*volume/(valloss/(i+1)))    # psnr(Y_pred,Y)
            valssim += ssim_(Y_pred,Y)

            # view results in tensorboard
            k = random.randint(0,bsize-5)
            gridim = torchvision.utils.make_grid(X[k:k+4,:,:,:])
            gridlb = torchvision.utils.make_grid(Y[k:k+4,:,:,:])
            gridpr = torchvision.utils.make_grid(Y_pred[k:k+4,:,:,:])
            writer.add_image('valimages', gridim, 0)
            writer.add_image('labels', gridlb, 0)
            writer.add_image('pred', gridpr, 0)         
            
    # Adjust metrics to get average loss and snr per batch 
    valloss   = valloss/len(dataloader)
    valpsnr   = valpsnr/len(dataloader)
    valpsnrlq = valpsnrlq/len(dataloader)
    valssim   = valssim/len(dataloader)
    return valloss, valpsnr, valpsnrlq, valssim

def train(args, device, tb_writer):
    # cache variables
    savedir, factor, stddev, quality, epochs, batchsize, imagesize, numworkers, resume, save, saveperiod, val, valperiod, task = \
        Path(args.savedir), args.factor, args.stddev, args.quality, args.epochs, args.batchsize, args.imgsize, args.workers, args.resume, \
            args.save, args.saveperiod, args.val, args.valperiod, args.task
    
    # Make directories
    wdir = savedir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last, best, resultsfile = wdir / 'last.pt', wdir / 'best.pt', savedir / 'results.txt'

    # Setup Dataloaders
    with open(args.data) as f:
        datadict = yaml.load(f, Loader=yaml.SafeLoader)

    traindir, valdir = datadict['train'], datadict['val']
    deblocking = True if task == 'car' else False
    datatrain = datagenerator(data_dir=traindir, patch_size=imagesize, batch_size=batchsize, is_gray=False, deblocking=deblocking, quality=quality)
    maxval = 255.0 if datatrain.max() > 2.0 else 1.0
    #maxval = datatrain.max()
    datatrain = datatrain.astype('float32')/maxval # if uint8 or 0 < val < 255 /255.0
    datatrain = torch.from_numpy(datatrain.transpose((0, 3, 1, 2)))
    #dataval = datagenerator(data_dir=valdir, patch_size=imagesize, batch_size=batchsize)
    #dataval = datatrain.astype('float32')  # if uint8 /255.0
    #dataval = torch.from_numpy(datatrain.transpose((0, 3, 1, 2)))

    dataval = datatrain
    if task=='sisr':
        trainset = SuperresolutionDataset(datatrain, factor)
        valset = SuperresolutionDataset(dataval, factor)
    elif task=='misr':
        trainset = MultiChannelSuperresolutionDataset(datatrain, factor)
        valset = MultiChannelSuperresolutionDataset(dataval, factor)
    elif task=='dn':
        trainset = DenoisingDataset(datatrain, stddev)
        valset = DenoisingDataset(dataval, stddev)
    elif task=='car':
        trainset = DeblockingDataset(datatrain)
        valset = DeblockingDataset(dataval)
    else:
        print("Select the correct task")
    trainloader = tdata.DataLoader(dataset=trainset, batch_size=batchsize, num_workers=numworkers,drop_last=True,shuffle=True)
    valloader = tdata.DataLoader(dataset=valset, batch_size=batchsize, num_workers=numworkers,drop_last=True,shuffle=False)

    # Training model and hyperparameters setup
    if task=='sisr' or task=='car':
        # model = DnCNN().to(device)
        model = CustomRegressionNetAtt(numlayers=9).to(device) #CustomRegressionNet(numlayers=15).to(device)
    elif task=='misr':
        model = MultChannelRegressionNetAtt(scale=factor,numlayers=6,numreslayers=4).to(device)
    elif task=='dn':
        model = CustomDenosingRegressionNetAtt(numlayers=12).to(device)

    images, labels = next(iter(trainloader))
    tb_writer.add_graph(model, images.to(device))

    # Calculate global batch stat and deactivate batchnorm running stats
    # calculate_batch_stats(model, trainloader, device)
    # deactivate_batchnorm_running(model)
    
    print(model)

    # if len(args.weights) > 0:
    #    model.load_state_dict(args.weights)

    # Optimizer
    bias_params, non_bias_params = getparams(model)
    if args.optim == 'adam':
        if not bias_params:
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = optim.Adam([
                {'params': non_bias_params},
                {'params': bias_params, 'lr': 1e-4},
            ], lr=1e-3)
    else:
        if not bias_params:
            optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=0.0001)
        else:
            optimizer = optim.SGD([
                {'params': non_bias_params},
                {'params': bias_params, 'lr': 1e-2, 'weight_decay': 1e-5},
            ], lr=1e-1, momentum=0.9, weight_decay=0.0001)

    # Define learning rate scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,50,80], gamma=0.7)

    # Define loss functions
    loss_fn = SSELoss().to(device) #nn.MSELoss(reduction='sum').to(device) # CustomRegressionLoss(1,1).to(device)
    loss_fn_perc = None
    loss_fn_dft = None # DFTLoss(2).to(device) if task=='car' else None
    loss_fn_ssim = SSIMLoss(1e-4).to(device)

    # Initialize results dictionary
    results = {"trainloss": [],
        "trainpsnr": [],
        "trainssim": [],
        "valloss": [],
        "valpsnr": [],
        "valssim": []
    }
    startepoch, oldvalpsnr = 0, -30.0

    if resume:
        ckpt = torch.load(args.weights, weights_only=False, map_location=device)
        if ckpt['model'] is not None:
            model.load_state_dict(ckpt['model'].state_dict())
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('results') is not None:
            results = ckpt['results']

        startepoch = ckpt['epoch'] + 1

    epochs_ = trange(startepoch,epochs)

    # Train
    for epoch in epochs_:
        trainstats = train_step(model=model,
                                device=device,
                                epochs_=epochs_,
                                epoch = epoch,
                                dataloader=trainloader,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                loss_fn_perc=loss_fn_perc,
                                loss_fn_dft=loss_fn_dft,
                                loss_fn_ssim=loss_fn_ssim)
        valstats = (0,0,0,0)
        final_epoch = epoch + 1 == epochs

        if val:
            if ((epoch+1) % valperiod) == 0 and (epoch+1) != epochs:
                valstats   = val_step(model=model,
                                    device=device,
                                    dataloader=valloader,
                                    loss_fn=loss_fn,
                                    writer=tb_writer)
                # Print training progress
                print(
                    f"Epoch: {epoch+1} | valloss: {valstats[0]:.4f} | valpsnr: {valstats[1]:.4f} | valpsnrlq: {valstats[2]:.4f} | valssim: {valstats[3]:.4f}"
                )               
        else:
            if final_epoch:
                valstats   = val_step(model=model,
                                    device=device,
                                    dataloader=valloader,
                                    loss_fn=loss_fn,
                                    writer=tb_writer)
                # Print training progress
                print(
                    f"Epoch: {epoch+1} | valloss: {valstats[0]:.4f} | valpsnr: {valstats[1]:.4f} | valpsnrlq: {valstats[2]:.4f} | valssim: {valstats[3]:.4f}"
                ) 
        
        # Update results dictionary and tensorboard writer
        update_stats(results, tb_writer, trainstats, valstats, epoch)

        # Schedule learning rate
        scheduler.step()

        # Save model
        if save:
            if ((epoch+1) % saveperiod) == 0 and (epoch+1) != epochs:
                torch.save({'model':model}, wdir / 'epoch_{:03d}.pt'.format(epoch))
        if val and ((epoch+1) % valperiod)==0:
            if valstats[1] > oldvalpsnr:
                torch.save({'model':model}, best)
            oldvalpsnr = valstats[1]
        
        # Save the model from the immediate epoch as last
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer.state_dict(),
            'results': results,
            }, last)
    
    # Write the results
    with open(resultsfile,'w') as f:
        f.write(json.dumps(results))
    tb_writer.close()

    print("All Results saved in ", savedir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='',help='select path of pretrained weights or saved weights')
    parser.add_argument('--data',type=str,default='data/traindataset.yaml',help='select the ml file for the appropriate training dataset')
    parser.add_argument('--task',type=str,default='dn',help='multiimage(misr),singleimage(sisr),denoising(dn),compressionartefactremoval(car)')
    parser.add_argument('--factor',type=int,default=2,help='select superresolution factor')
    parser.add_argument('--stddev',type=int,default=15,help='select denoising sigma')
    parser.add_argument('--quality',type=int,default=10,help='select quality factor compression artefact removal')
    parser.add_argument('--epochs',type=int,default=80,help='Number of Epochs')
    parser.add_argument('--batchsize', type=int, default=12, help='total batch size for all CPUs/GPUs')
    parser.add_argument('--imgsize', type=int, default=128, help='[train, val] image sizes')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or cpu, multi gpu support or DDP not implemented')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--resume', type=bool, default=False, help='resume most recent training')
    parser.add_argument('--save', type=bool, default=True, help='save at "saveperiod" intervals, else save the last epoch')
    parser.add_argument('--saveperiod', type=int, default=5, help='Log model after every "saveperiod" epoch')
    parser.add_argument('--val', type=bool, default=True, help='validate after every "valperiod" epochs, else only after the last epoch')
    parser.add_argument('--valperiod', type=int, default=2, help='number of epochs to validate after')
    parser.add_argument('--optim', type=str, default='adam', help='select optimizer')
    parser.add_argument('--project', type=str, default='runs/train', help='save to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--freeze', type=int, default=[0], help='Freeze layers: backbone of model, first3=0 1 2')
    args = parser.parse_args()

    if args.resume:
        ckpt = get_latest_run(search_dir=args.project,task=args.task)
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist, reset "resume" flag'
        args.weights, args.resume, args.batchsize, args.savedir = ckpt, True, args.batchsize, ckpt[0:-16]
        logger.info('Resuming training from %s' % ckpt)
    else:
        args.data = check_file(args.data)  # check files
        if len(args.weights):
            assert os.path.isfile(args.weights), 'ERROR: --weights path does not exist'
        args.savedir = increment_path(Path(args.project) / args.task / args.name, exist_ok=args.exist_ok)  # increment run

    device = select_device(args.device, batch_size=args.batchsize)

    # Train
    logger.info(args)
    tb_writer = None
    logger.info(f"{colorstr('tensorboard: ')}Start with 'tensorboard --logdir {args.savedir}', view at http://localhost:6006/")
    tb_writer = SummaryWriter(args.savedir)  # Tensorboard
    train(args, device, tb_writer)
