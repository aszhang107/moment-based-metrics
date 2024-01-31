#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:37:17 2021

@author: marcaurele
"""
import numpy as np
'''
Some simple scaling to map FT - > DFT and back, and handle frequencies in scaled units.
I suspect that all the scaling factors could be avoided when solving linear systems.
'''


def get_1d_frequency_grid(N, voxel_size = 1, scaled = False):
    if N % 2 == 0:
        grid =  np.linspace( - N/2, N/2 - 1 , N) 
    else:
        grid =  np.linspace( - (N - 1)/2, (N- 1)/2 , N)

    if scaled:
        grid = grid / ( N * voxel_size)
    
    return grid

def get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled = True):
    one_D_grids = [ get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape ]
    grids = np.meshgrid(*one_D_grids, indexing="ij")
    return np.transpose(np.vstack([np.reshape(g, -1) for g in grids]))

def get_grid_of_radial_distances(img_shape, voxel_size = 1, scaled = False):
    one_D_grids = [ get_1d_frequency_grid(sh, voxel_size, scaled)**2 for sh in img_shape ]
    grids = np.stack(np.meshgrid(*one_D_grids, indexing="ij"), axis =-1).astype(one_D_grids[0].dtype)    
    r = np.sqrt(np.sum(grids, axis = -1))
    
    if scaled:
        return r
    else:
        return np.round(r).astype(int)

def DFT_to_FT_scaling_vector(N, voxel_size):
    frequencies = get_1d_frequency_grid(N, voxel_size = 1, scaled = False)
    return np.exp(1j*np.pi*frequencies)

def DFT_to_FT_scaling(img_ft, voxel_size):
    N = img_ft.shape[0]
    scaling_vec = DFT_to_FT_scaling_vector(N, voxel_size)
    
    if img_ft.ndim ==1:
        phase_shifted_img_ft = (img_ft * scaling_vec) 
    if img_ft.ndim ==2:
        phase_shifted_img_ft = ((img_ft * scaling_vec[:,None]) * scaling_vec[None, :])
    if img_ft.ndim ==3:
        phase_shifted_img_ft = ((img_ft * scaling_vec[:,None,None]) * scaling_vec[None, :,None])* scaling_vec[None,None,:]
    return phase_shifted_img_ft
    

def FT_to_DFT_scaling(img_ft, voxel_size):
    N = img_ft.shape[0]
    scaling_vec = DFT_to_FT_scaling_vector(N, voxel_size)
    
    if img_ft.ndim ==1:
        phase_shifted_img_ft = (img_ft / scaling_vec) 
    if img_ft.ndim ==2:
        phase_shifted_img_ft = ((img_ft / scaling_vec[:,None]) / scaling_vec[None, :]) 
    if img_ft.ndim ==3:
        phase_shifted_img_ft = ((img_ft / scaling_vec[:,None,None]) / scaling_vec[None, :,None]) / scaling_vec[None,None,:]
    
    return phase_shifted_img_ft


def DFT_to_FT(img_dft, voxel_size):
    return DFT_to_FT_scaling(np.fft.fftshift(img_dft), voxel_size)
    
def FT_to_DFT(img_ft, voxel_size):
    return np.fft.ifftshift(FT_to_DFT_scaling(img_ft, voxel_size))

def get_grid_of_radial_distances(img_shape, voxel_size = 1, scaled = False):
    
    one_D_grids = [ get_1d_frequency_grid(sh, voxel_size, scaled)**2 for sh in img_shape ]
    grids = np.stack(np.meshgrid(*one_D_grids, indexing="ij"), axis =-1).astype(one_D_grids[0].dtype)    
    r = np.sqrt(np.sum(grids, axis = -1))
    if scaled:
        return r
    else:
        return np.round(r).astype(int)



def compute_index_dict(img_shape):
    r = get_grid_of_radial_distances(img_shape)
    from collections import defaultdict
    r_dict = defaultdict(list)
    for idx, ri in enumerate(r.flatten()):
        r_dict[ri].append(idx)    
    return r_dict


def compute_spherical_average_from_index_dict(img_ft, r_dict, use_abs = False):
    max_freq = np.max(list(r_dict.keys()))
    spherical_average = np.zeros(max_freq + 1, dtype = img_ft.dtype)
    img_ft_flat = img_ft.flatten()
    if use_abs:
        img_ft_flat = np.abs(img_ft_flat)
    for key, value in r_dict.items():
        spherical_average[key] = np.mean(img_ft_flat[value])
    return spherical_average


def compute_spherical_average(img_ft, r_dict = None, use_abs = False):
    r_dict = compute_index_dict(img_ft.shape) if r_dict is None else r_dict
    return compute_spherical_average_from_index_dict(img_ft, r_dict, use_abs = use_abs)


def get_fourier_transform(img, voxel_size):
    assert( not np.any(np.array(img.shape) ==1) )
    img_dft = np.fft.fftn(img )
    img_ft = DFT_to_FT(img_dft,voxel_size)
    return img_ft

def get_inverse_fourier_transform(img_ft, voxel_size):
    assert( not np.any(np.array(img_ft.shape) ==1) )
    img_dft = FT_to_DFT(img_ft,voxel_size)
    img_ft = np.fft.ifftn(img_dft )
    return img_ft

