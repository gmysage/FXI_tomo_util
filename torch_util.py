import torch
import numpy as np
from tqdm import trange
from image_util import otsu_mask_stack_mpi, img_smooth, crop_scale_image
from ml_model import RRDBNet, default_model_path, apply_model_to_stack

def apply_ML_prj(img, n_iter=1, filt_sz=1, model_path='', device='cuda'):

    if model_path == '':
        model_path = default_model_path(model_type='bkg')

    model_prod = RRDBNet(1, 1, 16, 4, 32).to(device)
    s = img.shape
    model_prod.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print('downscale image ...')
    img_rescale = crop_scale_image(img, output_size=(256, 256))
    img_output_scale, img_bkg_scale = apply_model_to_stack(img_rescale, model_prod, device, n_iter,
                                                                 gaussian_filter=filt_sz)
    print('upscale image...')
    img_bkg = crop_scale_image(img_bkg_scale, (s[1], s[2]))
    img_output = img / img_bkg
    return img_output

def ostu_mask_3D(img3D, filt_sz=3, iter=2, bins=128):
    s = img3D.shape
    if len(s) == 2:
        img3D = img3D[np.newaxis]
    img_m = otsu_mask_stack_mpi(img3D, filt_sz, iter, bins)
    return img_m


def apply_ML_tomo(img3D, model_name, model_path, device='cuda'):
    if len(img3D.shape) == 2:
        img3D = img3D[np.newaxis]
    s = img3D.shape
    img3D[img3D<0] = 0

    tmp = img3D[img3D>0]
    scale = np.sort(tmp)[int(len(tmp)*0.95)]
    if model_path == '':
        model_path = default_model_path(model_type='tomo')
    if 'RRDB 4' in model_name:
        model = RRDBNet(1, 1, 16, 4, 32)
    if 'RRDB 3' in model_name:
        model = RRDBNet(1, 1, 16, 3, 32)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except Exception as err:
        print(err)
        print('return raw images')
        return img3D, img3D
    model = model.to(device)
    img3D_m = img3D / scale
    img_d = np.zeros_like(img3D)
    with (torch.no_grad()):
        for i in trange(s[0]):
            img_torch = torch.from_numpy(img3D_m[i:i+1, np.newaxis]).float().to(device)
            t = model(img_torch)
            img_d[i] = t.cpu().numpy().squeeze() * scale
    return img_d


def medifilt_3D(img3D, filt_sz=3):
    img_d = img_smooth(img3D, filt_sz)
    return img_d

