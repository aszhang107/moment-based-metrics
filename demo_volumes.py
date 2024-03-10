import time
import pickle
import multiprocessing
import numpy as np
import prody
from itertools import combinations
import sys
sys.path.insert(0, './utils')
from moment_utils import read_pdb_ids_from_csv

FROM_MAP = True # adjust this if the volumes are taken from maps

if FROM_MAP:
    vol_id_list = ['000', '001', '002', '003', '004']
    pairwise_volumes = list(combinations(vol_id_list, 2))
else:
    pdb_file = 'pdb_id_list.txt' #file containing pdb ids
    pdb_id_list = read_pdb_ids_from_csv(pdb_file) 
    pairwise_volumes = list(combinations(pdb_id_list, 2))


N = 64 #grid size
voxel_size = 200 / N #voxel size in angstroms
phi_grid_size = N #grid of angles

save_path = '/scratch/gpfs/az8940' #where to save to
prody.pathPDBFolder('{}/pdbs'.format(save_path)) #directory with pdbs
num_workers = 16 #workers for parallel processing

r_image = np.fft.fftfreq(N, voxel_size)[N//2 - 1]
c_grid = np.linspace(0, r_image, N//2, endpoint=True)
phi_grid = np.linspace(0, 2*np.pi, phi_grid_size, endpoint=False)
r_size = len(c_grid)

def getMoments(vol, from_map = FROM_MAP):
    if from_map:
        with open('{}/uniform_moments_from_map/{}.pkl'.format(save_path, vol), 'rb') as handle:
            return pickle.load(handle)
    else:
        with open('{}/uniform_moments/{}.pkl'.format(save_path, vol), 'rb') as handle:
            return pickle.load(handle)
    
    
def volumeMetric(vol_pair):
    vol_1, vol_2 = vol_pair
    m1_1, m2_1 = getMoments(vol_1)
    moments_1 = np.concatenate((m1_1, (m2_1 * (np.expand_dims(np.sqrt(c_grid), 1) @ np.expand_dims(np.sqrt(c_grid), 0))).flatten()))
    m1_2, m2_2 = getMoments(vol_2)
    moments_2 = np.concatenate((m1_2, (m2_2 * (np.expand_dims(np.sqrt(c_grid), 1) @ np.expand_dims(np.sqrt(c_grid), 0))).flatten()))
    return np.linalg.norm(moments_1 - moments_2)**2

#Calculate pairwise distances
starttime = time.time()

with multiprocessing.Pool(num_workers) as pool:
    distances = pool.map(volumeMetric, pairwise_volumes)
print('Runtime: {} minutes'.format((time.time() - starttime)/60))