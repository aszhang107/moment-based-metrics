import time
import multiprocessing
import numpy as np
import pyshtools as pysh
import scipy.sparse
from aspire.volume import Volume
from fourier_transform_utils import get_inverse_fourier_transform
from generate_synthetic_molecule import generate_molecule_spectrum_from_pdb_id, center_atoms
from utils_BO import align_BO
from utils import read_pdb_ids_from_csv

pdb_file = 'pdb_id_list.txt'
pdb_id_list = read_pdb_ids_from_csv(pdb_file)
fixed_structure = '7VV3'
N = 64 #grid size
voxel_size = 200 / N #voxel size in angstroms
phi_grid_size = N #grid of angles


#Create list to compare to
pdb_id_list.remove(fixed_structure)

#Generate volume for fixed structure
fixed_vol_hat = generate_molecule_spectrum_from_pdb_id(fixed_structure, voxel_size, phi_grid_size)
fixed_vol = get_inverse_fourier_transform(fixed_vol_hat, voxel_size = voxel_size).real
fixed_vol_final = Volume(fixed_vol)

def align(pdb):
    #Generate volume for comparison structure
    vol2_hat = generate_molecule_spectrum_from_pdb_id(pdb, voxel_size, phi_grid_size)
    vol2 = get_inverse_fourier_transform(vol2_hat, voxel_size = voxel_size).real
    print("Alignment started for pdbs: {} and {}".format(fixed_structure, pdb))
    t1 = time.time()
    loss, R = align_BO(fixed_vol_final, Volume(vol2), ('eu', 32, 200, False),reflect=False)
    t2 = time.time()
    print("Time to compute alignment: {} seconds".format(t2 - t1))
    return pdb, loss, R

#Generate alignment scores and optimal rotations
alignment_scores = list(map(align, pdb_id_list))