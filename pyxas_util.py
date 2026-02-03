import pyxas
import torch
import numpy as np
from tqdm import trange

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

def apply_ML_tomo(img3D, model_path, device='cuda'):
    s = img3D.shape
    img3D[img3D<0] = 0
    img3D = pyxas.otsu_mask_stack(img3D, 3, 2, 128)
    tmp = img3D[img3D>0]

    scale = np.sort(tmp)[int(len(tmp)*0.95)]
    model = pyxas.RRDBNet(1, 1, 16, 4, 32)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    img3D = img3D / scale

    img_d = np.zeros_like(img3D)
    with (torch.no_grad()):
        for i in trange(s[0]):
            img_torch = torch.from_numpy(img3D[i:i+1, np.newaxis]).float().to(device)
            t = model(img_torch)
            img_d[i] = t.cpu().numpy().squeeze() * scale

    img_comb = np.concatenate((img3D*scale, img_d), axis=2)
    return img_d, img_comb

def masked_bkg_avg(img):
    _, mask, _ = pyxas.kmean_mask(img, 2)
    bkg = mask * img
    bkg_avg = np.sum(bkg) / np.sum(mask)
    return bkg_avg

def load_default_recon_model(device='cuda'):
    model_prod = pyxas.RRDBNet(1, 1, 16, 4, 32).to(device)

    fn_model_root = spec.submodule_search_locations[0] # e.g. '/data/FL_correction/FL'