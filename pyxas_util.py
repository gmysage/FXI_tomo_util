import pyxas
import torch


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

def masked_bkg_avg(img):
    _, mask, _ = pyxas.kmean_mask(img, 2)
    bkg = mask * img
    bkg_avg = np.sum(bkg) / np.sum(mask)
    return bkg_avg