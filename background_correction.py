# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:46:16 2024

@author: Alexandros Papagiannakis, HHMI at Stanford University
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import remove_small_objects


def get_common_divisors(x_dim, y_dim, verbose=False):
# common divisors for 140 and 420 dimensions that can be used to determine the square size during background estimation
    common_div_list = []
    if verbose:
        print(f'getting common divisors for the x dimensions of {x_dim} px and the y dimensions of {y_dim} px:')
    for i in range(10,np.max([x_dim, y_dim])):
        if y_dim%i==0 and x_dim%i==0:
            if verbose:
                print('x,y:',i)
            common_div_list.append(i)
    return common_div_list


def cell_free_bkg_estimation(masked_signal_image, step):
    """
    This function scans the image using squared regions of specified size (step) 
    and applies the average cell-free background fluorescence per region.
    This function is used in the self.back_sub() function.
    
    Parameters
    ----------
    masked_signal_image: 2D numpy array - the signal image were the cell pixels are annotated as 0 
                         and the non-cell pixels maintain their original grayscale values
    step: integer (should be a divisor or the square image dimensions) - the dimensions of the squared region where 
          the cell-free background fluorescence is averaged
            example: for an 2048x2048 image, 128 is a divisor and can be used as the size of the edge of the square 

    Returns
    -------
    A 2D numpy array where the cell-free average background is stored for each square region with specified step-size
    """
    
    sensor = masked_signal_image.shape
    
    zero_image = np.zeros(sensor) # initiated an empty image to store the average cell-free background
    
    for y in range(0, sensor[0], step):
        for x in range(0, sensor[1], step):
            # cropped_image = img_bkg_sig[y:(y+step), x:(x+step)]
            cropped_mask = masked_signal_image[y:(y+step), x:(x+step)]
#                mean_bkg = np.mean(cropped_mask[np.nonzero(cropped_mask)].ravel()) # get the mean of the non-zero pixels
#                mean_bkg = scipy.stats.mode(cropped_mask[cropped_mask!=0].ravel())[0][0] # get the mode of the non-zero pixels
            mean_bkg = np.nanmedian(cropped_mask[np.nonzero(cropped_mask)].ravel()) # get the mean of the non-zero pixels
            zero_image[y:(y+step), x:(x+step)] = mean_bkg # apply this mean fluorescence to the original empty image
                   
    return zero_image


def image_dilation(image, dilation_rounds):
    # dilate the masked phase images
    if dilation_rounds > 0:
        image_dil = ndimage.binary_dilation(image, iterations=dilation_rounds)
        image_dil = np.array(image_dil)
        return image_dil
    else:
        return image


def get_otsu_mask(phase_image):
    """returns a masked image using the inverted phase contrast image and the Otsu threshold

    Args:
        phase_image (2D numpy array): phase contrast image

    Returns:
        phase_mask: binary 2D numpy array
    """
    # invert the image and apply an otsu threshold to separate the dimmest 
    # (or inversely brightest pixels) which correspond to the cells
    inverted_phase_image = 1/phase_image
    inverted_threshold = threshold_otsu(inverted_phase_image.ravel())
    phase_mask = inverted_phase_image > inverted_threshold
    
    return phase_mask


def back_sub(signal_image, phase_mask, dilation, estimation_step, smoothing_sigma, show):
    """
    Subtracts an n_order second degree polynomial fitted to the non-cell pixels.
    The 2D polynomial surface is fitted to the non-cell pixels only.
        The order of the polynomial depends on whether there is uneven illumination or not
    The non-cell pixels are masked as thos below the otsu threshold estimated on the basis of the inverted phase image.
    
    Parameters
    ----------
    signal_image: numpy.array - the image to be corrected
    phase_mask: numpy.array - the inverted mask return by the get_inverted_mask_function
    dilation: non-negative integer - the number of dilation rounds for the cell mask
    estimation_step: positive_integer - the size of the square edge used for average background estimation. Must divide the dimensions of the image perfectly
    smoothing_sigma: non-negative integer - the smoothing factor of the cell free background
    show: binary - True if the user wants to visualize the 2D surface fit
    
    Returns
    -------
    [0] The background corrected image (2D numpy array) also corrected for uneven excitation
    [1] The background corrected image (2D numpy array) 
    [2] The background pixel intensities
    """
    if show:
        print('Subtracting background...')
    common_divisors = get_common_divisors(signal_image.shape[1], signal_image.shape[0])
    if estimation_step not in common_divisors:
        raise ValueError(f'The estimation step must be a common divisor of the x and y dimensions: {common_divisors}')
    # dilate the masked phase images
    phase_mask_dil = image_dilation(phase_mask, dilation)
    
    # mask the signal image, excluding the dilated cell pixels
    masked_signal_image = signal_image * (~phase_mask_dil).astype(int)
    
    fluor_pixels = masked_signal_image.ravel()
    
    if show == True:
        plt.figure()
        plt.title('Masked fluorescence image for background subtraction')
        plt.imshow(masked_signal_image, cmap='gray', vmin=np.percentile(fluor_pixels, 5), vmax=np.percentile(fluor_pixels, 95))
        plt.plot([10, 10+5/0.16], [120,120], color='white', linewidth=2)
        plt.text(10,135, '5 um', fontweight='bold', color='white')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    # The dimensions of the averaging square os the estimation_step
    img_bkg_sig = cell_free_bkg_estimation(masked_signal_image, estimation_step)
    
    if show == True:
        plt.figure()
        ax=plt.gca()
        plt.title('Background estimation')
        bim = ax.imshow(img_bkg_sig, cmap='gray')
        plt.plot([10, 10+5/0.16], [100,100], color='red', linewidth=2)
        plt.text(10,115, '5 um', fontweight='bold', color='red')
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(bim, cax=cax)
        cbar.set_label('Intensity (a.u.)', rotation=90, fontsize=14)
        plt.show()
    
    # Smooth the reconstructed background image, with the filled cell pixels.
    # img_bkg_sig = img_bkg_sig.astype(np.int16)
    img_bkg_sig = ndimage.gaussian_filter(img_bkg_sig, sigma=smoothing_sigma)
    
    if show == True:
        plt.figure()
        ax=plt.gca()
        plt.title('Smooth background estimation')
        bim = ax.imshow(img_bkg_sig, cmap='gray')
        plt.plot([10, 10+5/0.16], [100,100], color='red', linewidth=2)
        plt.text(10,115, '5 um', fontweight='bold', color='red')
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(bim, cax=cax)
        cbar.set_label('Intensity (a.u.)', rotation=90, fontsize=14)
        plt.show()
    
    norm_img_bkg_sig = img_bkg_sig/np.max(img_bkg_sig.ravel())
    # subtract the reconstructed background from the original signal image
    bkg_cor = (signal_image - img_bkg_sig)/norm_img_bkg_sig
    bkg_cor_2 = signal_image - img_bkg_sig
    # use this line if you want to convert negative pixels to zero
    # bkg_cor[bkg_cor<0]=0
    if show == True:
        plt.figure()
        ax=plt.gca()
        plt.title('Background corrected image\nalso linearly corrected for uneven excitation')
        bim = ax.imshow(bkg_cor, cmap='gray', vmin=50, vmax=200)
        plt.plot([10, 10+5/0.16], [100,100], color='red', linewidth=2)
        plt.text(10,115, '5 um', fontweight='bold', color='red')
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(bim, cax=cax)
        cbar.set_label('Intensity (a.u.)', rotation=90, fontsize=14)
        plt.show()
    
    return bkg_cor, bkg_cor_2, img_bkg_sig