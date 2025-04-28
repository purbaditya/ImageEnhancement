import os
import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import glob
import cv2
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr, rgb2gray
from skimage.transform import rescale
from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, uniform_, normal_, sparse_, constant_, orthogonal_
from PIL import Image, ImageOps

# Collection of several functions, including a few collected from various sources provided below and modified
# https://github.com/cszn/DnCNN
# https://github.com/cszn/KAIR/tree/master


# clip gradient by value for a better SGD convergence
def gradient_clipping(optimizer, value):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-value, value)

# separate bias and non-bias params
def getparams(model):
    bias_params = []
    non_bias_params = []
    for name, params in model.named_parameters():
        if 'bias' in name and params.requires_grad:
            bias_params.append(params)
        else:
            non_bias_params.append(params)
    return bias_params, non_bias_params

# activation / deactivation of batch norm running stats, calculation of global stats
def deactivate_batchnorm_running(model):
    for _ , m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m._saved_running_mean, m.running_mean = m.running_mean, None
            m._saved_running_var, m.running_var = m.running_var, None


def activate_batchnorm_running(model):
    for _ , m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.running_mean = m._saved_running_mean
            m.running_var = m._saved_running_var

def calculate_batch_stats(model, dataloader, device):
    with torch.inference_mode():
        for (images,_) in tqdm(dataloader):
            model(images.to(device))

# global initialization of model learnable parameters
def init_params(model, init=kaiming_normal_, gain=1):
    for _ , m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
            if init==xavier_normal_ or init==xavier_uniform_ or init==orthogonal_:
                init(m.weight, gain)
            elif init==kaiming_normal_ or init==kaiming_uniform_:
                init(m.weight, nonlinearity='relu')
            elif init==normal_:
                init(m.weight, mean=0, std=math.sqrt(2*gain/(math.prod(m.weight.shape)/m.weight.shape[1])))
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)        
        if isinstance(m, nn.BatchNorm2d):
            #torch.clamp(normal_(m.weight.data, 0.0, math.sqrt(gain*2./9./m.weight.shape[0])),-0.025,0.025)
            m.eps = 1e-3
            m.momentum = 0.05

# image size alteration
def im2torch(image):
    image = transforms.toTensor()(image) if not torch.is_tensor(image) else image
    if image.dim()==2:
        image = image.unsqueeze(0)
    return image

def shave(image, scale):
    image = im2torch(image)
    image = image[:,scale:image.shape[-2]-scale,scale:image.shape[-2]-scale]
    return image

def modcrop(image,scale):
    image = im2torch(image)
    croph, cropw = image.shape[-2]%scale, image.shape[-1]%scale
    image = image[:,0:image.shape[-2]-croph,0:image.shape[-1]-cropw]
    return image

def upscale(image,label):
        if image.shape[-1] < label.shape[-1]:
            scale = int(label.shape[-1]/image.shape[-1])
            image = torch.nn.functional.interpolate(image[:,0,:,:].unsqueeze(1), scale_factor=scale,mode='bicubic')
        return image

def generate_lossy_images(data_dir, data_dir_lossy, isgray, quality):
    if not os.path.isdir(data_dir_lossy):
        os.makedirs(data_dir_lossy)
    file_list = glob.glob(data_dir+'/*.png')
    for i in range(len(file_list)):
        image = imread(file_list[i], as_gray=isgray)
        file_name = file_list[i].split('\\')[-1].split('.')[0]
        #file_name = file_list[i].split('/')[-1].split('.')[0]
        if len(image.split()) == 3:
            image, _, _ = image.convert('YCbCr').split()
        file_ = data_dir_lossy+'/'+file_name+'.jpeg'
        if not os.path.isfile(file_):
            image.save(file_, format='jpeg', quality= quality, keep_rgb=False)

# generation of patch tensors
# modified from https://github.com/cszn/DnCNN
def data_augmentation(image, mode=0):
    # data augmentation
    if mode == 0:
        return image
    elif mode == 1:
        return np.flipud(image)
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3:
        return np.flipud(np.rot90(image))
    elif mode == 4:
        return np.rot90(image, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))


def generate_patches(file_name, patch_size, scales, aug_times, isgray, deblocking, quality):
    # get multiscale patches from a single image
    image = imread(file_name, as_gray=isgray)
    if len(image.shape)==3:
        if np.sum(np.abs(image[:,:,0]-image[:,:,1])) > 10:
            image = rgb2ycbcr(image)/255
        image = image[:,:,0]

    h, w = image.shape

    if deblocking:
        image_name = file_name.split('\\')[-1]
        # image_name = file_name.split('/')[-1]
        file_name_lq = file_name.replace('\\'+image_name,'Lossy/'+str(quality)+'\\'+image_name.replace('png','jpeg'))
        image_lq = Image.open(file_name_lq)
        if len(image_lq.split())<3:
            image_lq = np.expand_dims(np.asarray(image_lq)/255, axis=2)
        else:
            image_lq,_,_ = image_lq.convert('YCbCr').split()
            image_lq = np.expand_dims(np.asarray(image_lq)/255, axis=2)
        image = np.expand_dims(image, axis=2)
        image = np.concatenate((image_lq,image),axis=2)
        h, w, _ = image.shape

    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        #img_scaled = rescale(image, scale=s, order=3, anti_aliasing=True)
        if s < 1:
            img_scaled = cv2.resize(image, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        else:
            img_scaled = image

        if img_scaled.max() > 2:
            img_scaled = img_scaled.astype(np.float32)/255
        # extract patches
        for i in range(0, h_scaled-patch_size+1, int(1.8*patch_size)):
            for j in range(0, w_scaled-patch_size+1, int(1.8*patch_size)):
                if len(img_scaled.shape)>2:
                    x = img_scaled[i:i+patch_size, j:j+patch_size,:]
                else:
                    x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    if x.shape[:2]==(patch_size,patch_size):
                        x_aug = data_augmentation(x, mode=np.random.randint(0, 8))
                        patches.append(x_aug)
    return patches

def datagenerator(data_dir='datasets/enhancement/Train400', patch_size=40, batch_size=128, scales=[1], aug_times=1, is_gray=True, deblocking=False, quality=30, verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    if deblocking:
        file_list_lq = glob.glob(data_dir+'Lossy/'+str(quality)+'/*.png')
        if not file_list_lq:
            generate_lossy_images(data_dir, data_dir+'Lossy/'+str(quality), is_gray, quality)
    # initialize
    data = []
    # generate patches
    for i in range(0,len(file_list),1):
        patches = generate_patches(file_list[i], patch_size, scales, aug_times, is_gray, deblocking, quality)
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done')
    data = np.array(data, dtype='float32')
    if len(data.shape)<4:
        data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print(f"{data.shape[0]} training patches generated \n")
    return data

# https://github.com/cszn/KAIR/tree/master
# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
                (im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
                (im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = (
                (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
                (im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
                n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (
                n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def calculate_psnrb(img1, img2, border=0):
    """Calculate PSNR-B (Peak Signal-to-Noise Ratio).
    Ref: Quality assessment of deblocked images, for JPEG image deblocking evaluation
    # https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        img1, img2 = np.expand_dims(img1, 2), np.expand_dims(img2, 2)

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # follow https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.

    total = 0
    for c in range(img1.shape[1]):
        mse = torch.nn.functional.mse_loss(img1[:, c:c + 1, :, :], img2[:, c:c + 1, :, :], reduction='none')
        bef = _blocking_effect_factor(img1[:, c:c + 1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return float(total) / img1.shape[1]

# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# --------------------------------------------
# imresize for tensor image [0, 1]
# --------------------------------------------
def imresize(image, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: image: pytorch tensor, CHW or HW [0,1]
    # output: CHW or HW [0,1] w/o round
    need_squeeze = True if image.dim() == 2 else False
    if need_squeeze:
        image.unsqueeze_(0)
    in_C, in_H, in_W = image.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(image)

    sym_patch = image[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()
    return out_2


# --------------------------------------------
# imresize for numpy image [0, 1]
# --------------------------------------------
def imresize_np(image, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: image: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    image = torch.from_numpy(image)
    need_squeeze = True if image.dim() == 2 else False
    if need_squeeze:
        image.unsqueeze_(2)

    in_H, in_W, in_C = image.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(image)

    sym_patch = image[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = image[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()