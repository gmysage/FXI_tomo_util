from multiprocessing import Pool, cpu_count
from tqdm import tqdm, trange
from scipy.signal import medfilt2d
from skimage.filters import threshold_otsu
from scipy import ndimage
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

def _otsu_worker(args):
    img_slice, kernal_size, iters, bins, erosion_iter = args
    return otsu_mask(img_slice, kernal_size, iters, bins, erosion_iter)

def otsu_mask(img, kernal_size, iters=1, bins=256, erosion_iter=0):
    img_s = img.copy()
    img_s[np.isnan(img_s)] = 0
    img_s[np.isinf(img_s)] = 0
    for i in range(iters):
        img_s = img_smooth(img_s, kernal_size)
    thresh = threshold_otsu(img_s, nbins=bins)
    mask = np.zeros(img_s.shape)
    #mask = np.float32(img_s > thresh)
    mask[img_s > thresh] = 1
    mask = np.squeeze(mask)
    if erosion_iter:
        struct = ndimage.generate_binary_structure(2, 1)
        struct1 = ndimage.iterate_structure(struct, 2).astype(int)
        mask = ndimage.binary_erosion(mask, structure=struct1).astype(mask.dtype)
    mask[:erosion_iter+1] = 1
    mask[-erosion_iter-1:] = 1
    mask[:, :erosion_iter+1] = 1
    mask[:, -erosion_iter-1:] = 1
    return mask

def otsu_mask_stack(img, kernal_size, iters=1, bins=256, erosion_iter=0):
    s = img.shape
    img_m = np.zeros(s)
    for i in trange(s[0]):
        img_m[i] = otsu_mask(img[i], kernal_size, iters, bins, erosion_iter)
    img_r = img * img_m
    return img_r

def otsu_mask_stack_mpi(img, kernal_size, iters=1, bins=256, erosion_iter=0, n_processes=None):
    if n_processes is None:
        n_processes = cpu_count() // 2
    s = img.shape
    # Prepare arguments for each slice
    args = [
        (img[i], kernal_size, iters, bins, erosion_iter)
        for i in range(s[0])
    ]
    # Parallel processing
    with Pool(processes=n_processes) as pool:
        masks = list(pool.imap(_otsu_worker, args))
    # Rebuild mask stack
    img_m = np.zeros(s)
    for i in range(s[0]):
        img_m[i] = masks[i]
    img_r = img * img_m
    return img_r

def img_smooth(img, kernal_size, axis=0):
    s = img.shape
    if len(s) == 2:
        img_stack = img.reshape(1, s[0], s[1])
    else:
        img_stack = img.copy()

    if axis == 0:
        for i in range(img_stack.shape[0]):
            img_stack[i] = medfilt2d(img_stack[i], kernal_size)
    elif axis == 1:
        for i in range(img_stack.shape[1]):
            img_stack[:, i] = medfilt2d(img_stack[:,i], kernal_size)
    elif axis == 2:
        for i in range(img_stack.shape[2]):
            img_stack[:, :, i] = medfilt2d(img_stack[:,:, i], kernal_size)
    return img_stack

def plot3D(data, axis=0, index_init=None):
    fig, ax = plt.subplots()
    if index_init is None:
        index_init = int(data.shape[axis] // 2)
    im = ax.imshow(data.take(index_init, axis=axis))
    fig.subplots_adjust(bottom=0.15)
    axslide = fig.add_axes([0.1, 0.03, 0.8, 0.03])
    im_slider = Slider(
        ax=axslide,
        label='index',
        valmin=0,
        valmax=data.shape[axis] - 1,
        valstep=1,
        valinit=index_init,
    )
    def update(val):
        im.set_data(data.take(val, axis=axis))
        fig.canvas.draw_idle()
    im_slider.on_changed(update)
    plt.show()
    return im_slider