import pyxas
import torch
import numpy as np
from tqdm import trange
from image_util import otsu_mask_stack,otsu_mask_stack_mpi

def apply_ML_prj(img, n_iter=1, filt_sz=1, model_path='', device='cuda'):
    if model_path == '':
        model_path = '/home/mingyuan/Work/pyxas/pyxas/pyml/trained_model/pre_traind_model_xanes_denoise.pth'

    model_prod = pyxas.RRDBNet(1, 1, 16, 4, 32).to(device)

    s = img.shape

    model_prod.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print('downscale image ...')
    img_rescale = pyxas.crop_scale_image(img, output_size=(256, 256))
    img_output_scale, img_bkg_scale = pyxas.apply_model_to_stack(img_rescale, model_prod, device, n_iter,
                                                                 gaussian_filter=filt_sz)
    print('upscale image...')
    img_bkg = pyxas.crop_scale_image(img_bkg_scale, (s[1], s[2]))
    img_output = img / img_bkg
    return img_output

def ostu_mask_3D(img3D, filt_sz=3, iter=2, bins=128):
    s = img3D.shape
    if len(s) == 2:
        img3D = img3D[np.newaxis]
    #img_m = pyxas.otsu_mask_stack(img3D, filt_sz, iter, bins)
    img_m = otsu_mask_stack_mpi(img3D, filt_sz, iter, bins)
    return img_m


def apply_ML_tomo(img3D, model_path, filt_param={}, device='cuda'):
    s = img3D.shape
    img3D[img3D<0] = 0

    if len(filt_param) == 3:
        filt_sz = filt_param['sz']
        filt_iter = filt_param['iter']
        filt_bins = filt_param['bins']
        #img3D_m = pyxas.otsu_mask_stack(img3D, filt_sz, filt_iter, filt_bins)
        img3D_m = otsu_mask_stack_mpi(img3D, filt_sz, filt_iter, filt_bins)
    else:
        img3D_m = img3D

    tmp = img3D_m[img3D_m>0]

    scale = np.sort(tmp)[int(len(tmp)*0.95)]
    model = pyxas.RRDBNet(1, 1, 16, 4, 32)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    img3D_m = img3D_m / scale

    img_d = np.zeros_like(img3D)
    with (torch.no_grad()):
        for i in trange(s[0]):
            img_torch = torch.from_numpy(img3D_m[i:i+1, np.newaxis]).float().to(device)
            t = model(img_torch)
            img_d[i] = t.cpu().numpy().squeeze() * scale

    img_comb = np.concatenate((img3D, img_d), axis=2)
    return img_d, img_comb

def masked_bkg_avg(img):
    _, mask, _ = pyxas.kmean_mask(img, 2)
    bkg = mask * img
    bkg_avg = np.sum(bkg) / np.sum(mask)
    return bkg_avg

