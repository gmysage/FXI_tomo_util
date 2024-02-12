import tomopy
import numpy as np
import h5py
import time
from tqdm import tqdm, trange
import skimage.restoration as skr
from skimage.filters import gaussian as gf
try:
    from pyxas_util import *
    import pyxas
    exist_pyxas = True
except:
    exist_pyxas = False
def rotcen_test(fn,
                attr_proj='img_tomo',
                attr_flat='img_bkg',
                attr_dark='img_dark',
                attr_angle='angle',
                sli_start=None, sli_stop=None, sli_steps=None, sli=0,
                block_list=[], denoise_flag=0,
                algorithm='gridrec',
                n_iter=5,
                circ_mask_ratio=0.95,
                options={},
                dark_scale=1,
                snr=3,
                fw_level=9,
                filter_name='None',
                ml_param = {}
                ):
    f = h5py.File(fn, "r")
    tmp = np.array(f[attr_proj][0])
    s = [1, tmp.shape[0], tmp.shape[1]]

    if denoise_flag:
        addition_slice = 100
    else:
        addition_slice = 0

    if sli == 0:
        sli = int(s[1] / 2)
    sli_exp = [
        np.max([0, sli - addition_slice // 2]),
        np.min([sli + addition_slice // 2 + 1, s[1]]),
    ]

    theta = np.array(f[attr_angle]) / 180.0 * np.pi
    img_tomo = np.array(f[attr_proj][:, sli_exp[0]: sli_exp[1], :])

    img_dark = np.array(f[attr_dark][:, sli_exp[0]: sli_exp[1], :])
    img_dark = np.median(img_dark, axis=0, keepdims=True)

    img_bkg = np.array(f[attr_flat][:, sli_exp[0]: sli_exp[1], :])
    img_bkg = np.median(img_bkg, axis=0, keepdims=True)
    f.close()

    prj = (img_tomo - img_dark / dark_scale) / (img_bkg - img_dark / dark_scale)
    prj = ml_denoise(prj, ml_param)

    prj_norm = -np.log(prj)
    prj_norm = denoise(prj_norm, denoise_flag)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0

    s = prj_norm.shape
    if len(s) == 2:
        prj_norm = prj_norm.reshape(s[0], 1, s[1])
        s = prj_norm.shape

    if theta[-1] > theta[1]:
        pos = find_nearest(theta, theta[0] + np.pi)
    else:
        pos = find_nearest(theta, theta[0] - np.pi)
    block_list = list(block_list) + list(np.arange(pos + 1, len(theta)))
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if snr > 0:
        prj_norm = tomopy.prep.stripe.remove_all_stripe(prj_norm, snr=snr)
    if fw_level > 0:
        prj_norm = tomopy.prep.stripe.remove_stripe_fw(prj_norm, level=fw_level)

    if sli_start <= 0:
        sli_start = int(s[2] / 2 - 30)
        sli_stop = int(s[2] / 2 + 30)
        sli_steps = 30
    cen = np.linspace(sli_start, sli_stop, sli_steps, endpoint=False)
    img = np.zeros([len(cen), s[2], s[2]])
    for i in range(len(cen)):
        if 1:
            print("{}: rotcen {}".format(i + 1, cen[i]))
            if algorithm == 'gridrec':
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice // 2: addition_slice // 2 + 1],
                    theta,
                    center=cen[i],
                    algorithm="gridrec",
                    filter_name=filter_name
                )
            elif 'astra' in algorithm:
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice // 2: addition_slice // 2 + 1],
                    theta,
                    center=cen[i],
                    algorithm=tomopy.astra,
                    options=options
                )
            else:
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice // 2: addition_slice // 2 + 1],
                    theta,
                    center=cen[i],
                    algorithm=algorithm,
                    num_iter=n_iter,
                    # filter_name=filter_name
                )
    img = tomopy.circ_mask(img, axis=0, ratio=circ_mask_ratio)
    return img, cen, sli_start, sli_stop, sli_steps, sli_exp[0]


def recon_and_save(fn,
                  rot_cen,
                  attr_proj='img_tomo',
                  attr_flat='img_bkg',
                  attr_dark='img_dark',
                  attr_angle='angle',
                  sli=[],
                  binning=None,
                  block_list=[],
                  dark_scale=1,
                  denoise_flag=0,
                  snr=1,
                  fw_level=9,
                  algorithm='gridrec',
                  circ_mask_ratio=0.95,
                  fsave_flag = True,
                  fsave_root = '.',
                  fsave_prefix = '',
                  roi_cen = [],
                  roi_size = [],
                  return_flag = True,
                  ml_param = {}
                  ):
    print('Loading imaging data ... ')
    f = h5py.File(fn, "r")
    tmp_tomo = np.array(f[attr_proj][0:1])

    slice_info = ""
    bin_info = f"_bin_{binning}" if binning else 1
    s = tmp_tomo.shape
    if len(sli) == 0:
        sli = [0, s[1]]
    elif len(sli) == 1 and sli[0] >= 0 and sli[0] <= s[1]:
        if sli[0] == 0:
            sli = [s[1]//2, s[1]//2+1]
        sli = [sli[0], sli[0] + 1]
        slice_info = f"_slice_{sli[0]}"
    elif len(sli) == 2 and sli[0] >= 0 and sli[1] <= s[1]:
        slice_info = f"_slice_{sli[0]}_{sli[1]}"
    else:
        print("non valid slice id, will take reconstruction for the whole object")
    img_tomo = np.array(f[attr_proj][:, sli[0]:sli[1]])

    xeng = np.array(f['X_eng']) if 'X_eng' in f else 0
    scan_id = np.array(f['scan_id']) if 'scan_id' in f else 0
    angle_list = np.array(f[attr_angle])

    img_dark = np.array(f[attr_dark][:, sli[0]:sli[1]])
    if len(img_dark.shape) == 3:
        img_dark = np.median(img_dark, axis=0, keepdims=True)
    img_bkg = np.array(f[attr_flat][:, sli[0]:sli[1]])
    if len(img_bkg.shape) == 3:
        img_bkg = np.median(img_bkg, axis=0, keepdims=True)
    f.close()

    proj0 = (img_tomo - img_dark / dark_scale) / (img_bkg - img_dark / dark_scale)
    proj0[np.isinf(proj0)] = 0
    proj0[np.isnan(proj0)] = 0
    proj0[proj0<0] = 0
    rec = recon_img(proj0, angle_list, rot_cen, binning, block_list,
                    denoise_flag, snr, fw_level, algorithm, circ_mask_ratio, ml_param)
    s1 = rec.shape # (400, 1280, 1280)
    if len(roi_cen) == 2 and len(roi_size) == 2:
        roi_cen = np.array(roi_cen) // binning
        roi_size = np.array(roi_size) // binning
        r_s = max(0, int(roi_cen[0]-roi_size[0]/2))
        r_e = min(s1[1], r_s + int(roi_size[0]))
        c_s = max(0, int(roi_cen[1] - roi_size[1] / 2))
        c_e = min(s1[2], c_s + int(roi_size[1]))
        rec = rec[:, r_s:r_e, c_s:c_e]

    if len(fsave_prefix) == 0:
        tmp = fn.split('/')[-1]
        tmp = tmp.split('.')[0]
        tmp1 = tmp.split('_')[-1]
        fsave_prefix = tmp1 if len(tmp1) else tmp
    if fsave_root[-1] == '/':
        fsave_root = fsave_root[:-1]
    fsave = f"{fsave_root}/recon_{fsave_prefix}{slice_info}{bin_info}.h5"

    if fsave_flag:
        print('saving data ...')
        with h5py.File(fsave, "w") as hf:
            hf.create_dataset("img", data=rec)
            hf.create_dataset("rot_cen", data=rot_cen)
            hf.create_dataset("binning", data=binning)
            hf.create_dataset("scan_id", data=scan_id)
            hf.create_dataset("X_eng", data=xeng)
        print(f'file saved to {fsave}')
    if return_flag:
        return rec, fsave


def recon_img(proj0, angle_list, rot_cen, binning=None, block_list=[], denoise_flag=0, snr=0,
              fw_level=0, algorithm='gridrec', circ_mask_ratio=0.95, ml_param={}):
    ts = time.time()
    tmp = proj0[0]
    s = [1, tmp.shape[0], tmp.shape[1]]


    theta = angle_list / 180.0 * np.pi
    rot_cen = (rot_cen * 1.0) / binning

    img_norm = bin_image_stack(proj0, binning)
    img_norm = denoise(img_norm, denoise_flag)

    n_angle = len(theta)
    total_id = np.arange(n_angle)
    if len(block_list):
        idx = set(list(total_id)) - set(list(block_list))
        idx = np.sort(list(idx))
        img_norm = img_norm[idx]
        theta = theta[idx]
    img_norm = ml_denoise(img_norm, ml_param)
    proj = -np.log(img_norm)
    proj[np.isnan(proj)] = 0
    proj[np.isinf(proj)] = 0
    '''
    if snr > 0:
        print('removing all stripe ...')
        proj = tomopy.prep.stripe.remove_all_stripe(proj, snr=snr)
        #proj = tomopy.prep.stripe.remove_stripe_based_filtering(proj, sigma=3)
        #proj = tomopy.prep.stripe.remove_stripe_based_sorting(proj)
    if fw_level > 0:
        print('removing all stripe ...')
        proj = tomopy.prep.stripe.remove_stripe_fw(proj, level=fw_level)
    '''
    ts1 = time.time()

    print('reconstruction ...')
    extra_options = {'MinConstraint': 0, }
    options = {'proj_type': 'cuda',
               'method': 'FBP_CUDA',
               'num_iter': 20,
               'extra_options': extra_options
               }

    s = proj.shape  # e.g, (600, 1080, 1280)
    n_sli = 40
    n_step = s[1] // n_sli
    #idx_sli_remain = (n_step * n_sli, s[1])
    recon = np.zeros((s[1], s[2], s[2]))
    for i in tqdm(np.arange(n_step + 1), total=n_step):
        id_s = i * n_sli
        id_e = min((i + 1) * n_sli, s[1])
        if id_s >= id_e:
            break
        prj_sub = proj[:, id_s: id_e]
        if snr > 0:
            prj_sub = tomopy.prep.stripe.remove_all_stripe(prj_sub, snr=snr)
        if fw_level > 0:
            prj_sub = tomopy.prep.stripe.remove_stripe_fw(prj_sub, level=fw_level)
        prj_sub = denoise(prj_sub, denoise_flag)
        if 'astra' in algorithm:
            rec_sub = tomopy.recon(prj_sub,
                                     theta,
                                     center=rot_cen,
                                     algorithm=tomopy.astra,
                                     options=options,
                                     ncore=4)
        else:
            rec_sub = tomopy.recon(prj_sub, theta, center=rot_cen, algorithm='gridrec')
        recon[id_s:id_e] = rec_sub
    ts2 = time.time()
    del img_norm, proj
    recon = tomopy.circ_mask(recon, axis=0, ratio=circ_mask_ratio)
    print(f'time for loading data:   {ts1 - ts:3.2f} sec')
    print(f'time for reconstruction: {ts2 - ts1:3.2f} sec')
    return recon

def denoise(prj, denoise_flag):
    if denoise_flag == 1:  # Wiener denoise
        ss = prj.shape
        if ss[1] == 1: # single slice
            prj = np.ones((ss[0], 3, ss[-1])) * prj
        psf = np.ones([2, 2]) / (2**2)
        reg = None
        balance = 0.3
        is_real = True
        clip = True
        for j in range(ss[0]):
            prj[j] = skr.wiener(prj[j], psf=psf, reg=reg, balance=balance, is_real=is_real, clip=clip)
        if ss[1] == 1:
            prj = prj[:, 0:1]
    elif denoise_flag == 2:  # Gaussian denoise
        prj = gf(prj, [0, 1, 1])
    return prj


def ml_denoise(prj, ml_param):
    if exist_pyxas and len(ml_param):
        n_iter = ml_param.get('n_iter')
        filt_sz = ml_param.get('filt_sz')
        model_path = ml_param.get('model_path')
        device = ml_param.get('device')
        try:
            print('ml_denoise ...')
            prj_ml = apply_ML_prj(prj, n_iter, filt_sz, model_path, device=device)
            return prj_ml
        except Exception as err:
            print(err)
            return prj
    else:
        return prj

def bin_image_stack(img_stack, binning=1):
    if binning == 1:
        return img_stack
    s = img_stack.shape
    img_b = bin_ndarray(img_stack, (s[0], s[1]//binning, s[2]//binning))
    return img_b


def bin_ndarray(ndarray, new_shape=None, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if new_shape == None:
        s = np.array(ndarray.shape)
        s1 = np.int32(s/2)
        new_shape = tuple(s1)
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def find_nearest(data, value):
    data = np.array(data)
    return np.abs(data - value).argmin()