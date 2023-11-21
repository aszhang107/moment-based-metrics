import time
import pickle
import multiprocessing
import numpy as np
import pyshtools as pysh
import prody
import scipy.special as sp
from fourier_transform_utils import get_inverse_fourier_transform
from generate_synthetic_molecule import generate_potential_at_freqs_from_atoms
from aspire.volume import Volume
from aspire.utils.coor_trans import grid_3d, grid_2d

import finufft

# Here we define the grids used by pyshtools
def make_DH_grid_in_polar(Nsph):
    long = np.linspace(0, 2 *np.pi, Nsph, endpoint=False)
    lat = np.linspace(0, np.pi, Nsph, endpoint=False)
    long2, lat2 = np.meshgrid(long, lat)
    return long2.reshape(-1), lat2.reshape(-1)


def pol2cart(longlatr):
    long,lat,r = np.split(longlatr,3, axis =-1)
    x = np.cos(long) * np.sin(lat)
    y = np.sin(long) * np.sin(lat)
    z = np.cos(lat)
    return np.concatenate([x, y, z], axis = -1) #x,y,z


def cart2pol(xyz):
    x,y,z = np.split(xyz,3, axis =-1)

    r = np.linalg.norm(xyz, axis =-1)[...,None]
    lat = np.arccos(z / r)
    long = np.arctan2(y,x)
    return np.concatenate([long, lat, r], axis = -1) #long, lat, r


def make_DH_grid_in_cartesian(Nsph):
    long, lat = make_DH_grid_in_polar(Nsph)
    x = np.cos(long) * np.sin(lat)
    y = np.sin(long) * np.sin(lat)
    z = np.cos(lat)
    xyz = np.stack([x,y,z], axis = -1)
    return xyz


def compute_sph_coeffs(atom_group, voxel_size, grid_size, shell_radius):
    shell = shell_radius * make_DH_grid_in_cartesian(grid_size)
    X_ims = generate_potential_at_freqs_from_atoms(atom_group, voxel_size, shell).reshape(grid_size,grid_size)
    protein_coeffs = pysh.shtools.SHExpandDHC(X_ims)
    return protein_coeffs


def compute_all_sph_coeffs(atom_group, voxel_size, grid_size, shell_grid):
    shells = (shell_grid[...,None,None] * make_DH_grid_in_cartesian(grid_size)[None])

    shells_flat = shells.reshape(-1, 3)
    X_ims = generate_potential_at_freqs_from_atoms(atom_group, voxel_size, shells_flat)#.reshape(grid_size,grid_size)
    X_ims= X_ims.reshape(shell_grid.size, grid_size,grid_size)

    protein_coeffs = [pysh.shtools.SHExpandDHC(Xi) for Xi in X_ims ] 
    return protein_coeffs

def compute_all_sph_coeffs_from_map(file, voxel_size, grid_size, shell_grid, N, r_image):
    shells = (shell_grid[...,None,None] * make_DH_grid_in_cartesian(grid_size)[None])

    shells_flat =  (N - 2) / N * shells.reshape(-1, 3) * np.pi / r_image / 4
    if isinstance(file, str):
        vol = Volume.load(file).asnumpy()[0]
    else:
        vol = file
    vol = vol.astype(np.complex128)
    X_ims = finufft.nufft3d2(shells_flat[:,0], shells_flat[:,1], shells_flat[:,2], vol)
    X_ims= X_ims.reshape(shell_grid.size, grid_size,grid_size)

    protein_coeffs = [pysh.shtools.SHExpandDHC(Xi) for Xi in X_ims ] 
    return protein_coeffs


def get_sph_coeffs(shell_grid, atom_group, voxel_size, Nsph):
    # Shell grid is the grid of points r_1, r_2,... r_c at which we want to evaluate 
    # Atom group defines the particular molecule
    # voxel size is something that is given

    # radius x l x m
    SPH_coeffs2 = np.zeros([shell_grid.size, Nsph//2, Nsph  ], dtype = np.complex128)

    x2 = compute_all_sph_coeffs(atom_group, voxel_size, Nsph, shell_grid)

    for shell_idx, shell_radius in enumerate(shell_grid):

        SPH_coeffs2[shell_idx,:,:Nsph//2] = x2[shell_idx][0]
        SPH_coeffs2[shell_idx,:,Nsph//2:] = x2[shell_idx][1]

    return SPH_coeffs2


def get_sph_coeffs_from_map(shell_grid, file, voxel_size, Nsph, N, r_image):
    # Shell grid is the grid of points r_1, r_2,... r_c at which we want to evaluate 
    # Atom group defines the particular molecule
    # voxel size is something that is given

    # radius x l x m
    SPH_coeffs2 = np.zeros([shell_grid.size, Nsph//2, Nsph  ], dtype = np.complex128)

    x2 = compute_all_sph_coeffs_from_map(file, voxel_size, Nsph, shell_grid, N, r_image)

    for shell_idx, shell_radius in enumerate(shell_grid):

        SPH_coeffs2[shell_idx,:,:Nsph//2] = x2[shell_idx][0]
        SPH_coeffs2[shell_idx,:,Nsph//2:] = x2[shell_idx][1]

    return SPH_coeffs2


def compute_A_matrix(shell_grid, atom_group, voxel_size, Nsph):
    SPH_coeffs = get_sph_coeffs(shell_grid, atom_group, voxel_size, Nsph)
    M1 = SPH_coeffs[:,0,0] 
    
    # l x radius x m
    SPH_coeffs = SPH_coeffs.transpose(1,0,2)
    As = SPH_coeffs @ np.conj(SPH_coeffs.transpose(0,2,1))
    # l x radius x radius
    
    return M1, As, SPH_coeffs

def compute_A_matrix_from_map(shell_grid, file, voxel_size, Nsph, N, r_image):
    SPH_coeffs = get_sph_coeffs_from_map(shell_grid, file, voxel_size, Nsph, N, r_image)
    M1 = SPH_coeffs[:,0,0] 
    
    # l x radius x m
    SPH_coeffs = SPH_coeffs.transpose(1,0,2)
    As = SPH_coeffs @ np.conj(SPH_coeffs.transpose(0,2,1))
    # l x radius x radius
    
    return M1, As, SPH_coeffs


def scaled_chebyshev_points(n, radius):
    """returns roots from Chebyshev polynomial of degree n, scaled from 0 to radius"""
    
    c = np.polynomial.chebyshev.chebpts1(n)
    c_scaled = (c*radius)/2 + radius/2
    
    return c_scaled


def evaluate_legendre_polynomial(l, phi):
    """evaluate Legendre polynomial of dregree l at grid points phi"""
    
    return sp.eval_legendre(l, np.cos(phi))


def compute_moments(atom_group, Nsph, phi_grid, c_grid, voxel_size):
    
    M1, As, SPH_coeffs = compute_A_matrix(c_grid, atom_group, voxel_size, Nsph)

    Pl = np.array([evaluate_legendre_polynomial(l, phi_grid) for l in range(1, Nsph//2)]) 
    M2 = np.zeros((len(phi_grid), c_grid.size, c_grid.size), dtype='complex')

    for phi in range(len(phi_grid)): 
        M2[phi] = np.sum(As[1:,:,:] * Pl[:, phi].reshape(Nsph//2 - 1, 1, 1), axis=0) 
        
    return M1, M2, SPH_coeffs

def compute_moments_from_map(file, Nsph, phi_grid, c_grid, voxel_size, N, r_image):
    
    M1, As, SPH_coeffs = compute_A_matrix_from_map(c_grid, file, voxel_size, Nsph, N, r_image)

    Pl = np.array([evaluate_legendre_polynomial(l, phi_grid) for l in range(1, Nsph//2)]) 
    M2 = np.zeros((len(phi_grid), c_grid.size, c_grid.size), dtype='complex')

    for phi in range(len(phi_grid)): 
        M2[phi] = np.sum(As[1:,:,:] * Pl[:, phi].reshape(Nsph//2 - 1, 1, 1), axis=0) 
        
    return M1, M2, SPH_coeffs

class Molecule:
    
    def __init__(self, pdb_id, n_atoms, M1, M2): 
        
        self.pdb_id = pdb_id
        self.n_atoms = n_atoms
        self.M1 = M1
        self.M2 = M2 
      
    
def save_molecule_moments(filename, pdb_moments):
    with open(filename, 'wb') as f:
        pickle.dump(pdb_moments, f)

        
def read_molecule_moments(filename):
    with open(filename, 'rb') as f:
        pdb_moments = pickle.load(f)
        

# #make these better later
def read_pdb_ids_from_json(file):
    """outputs all pdb ids from json file into a list"""
    with open(file, 'r') as f:
        for line in f:
            line.rstrip('\n')
            pdb_id_list = line.split(',')
    pdb_id_list = [pdb_id.split("\"")[1] for pdb_id in pdb_id_list]
    return pdb_id_list


def read_pdb_ids_from_csv(file):
    """outputs all pdb ids from csv file into a list"""
    with open(file, 'r') as f:
        for line in f:
            line.rstrip('\n')
            pdb_id_list = line.split(',')  
    return pdb_id_list

def make_one_atom_atom_group():
    atom_coor = np.zeros((1,3))
    N_atoms = 1
    ATOM_TYPE = 'C'
    ATOMGROUP_FLAGS = ['hetatm', 'pdbter', 'ca', 'calpha', 'protein', 'aminoacid', 'nucleic', 'hetero']

    atoms = prody.AtomGroup()
    atoms.setCoords(atom_coor)
    atoms.setNames(np.array(N_atoms *[ATOM_TYPE],  dtype='<U6'))
    atoms.setElements(np.array(N_atoms *[ATOM_TYPE],  dtype='<U6'))

    atoms._flags = {}
    for key in ATOMGROUP_FLAGS:
        atoms._flags[key] = np.zeros(N_atoms, dtype = bool)
    return atoms

def rot_matrix_around_axis(vector, angle):
    x = vector[:, 0]
    y = vector[:, 1]
    z = vector[:, 2]
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos + (x**2)*(1 - cos), x*y*(1 - cos) - z*sin, x*z*(1 - cos) + y*sin],
                     [y*x*(1 - cos) + z*sin, cos + (y**2)*(1 - cos), y*z*(1 - cos) - x*sin],
                     [z*x*(1 - cos) - y*sin, z*y*(1 - cos) + x*sin, cos + (z**2)*(1 - cos)]]).swapaxes(0, 2).swapaxes(1, 2)

def generate_mixture_vonMises(means, covs, pis):
    """
    Inputs: means means of Gaussians
            covs covariance matrices of Gaussians
            pis probabilities of choosing each Gaussian
    Output: a random vector on the unit sphere according to mixture of vonMises distritions
    """
    means = means
    covs = covs
    pis = pis
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis)+1)]
    assert np.isclose(acc_pis[-1], 1)
    
    r = np.random.uniform(0, 1)
    # select Gaussian
    k = 0
    for i, threshold in enumerate(acc_pis):
        if r < threshold:
            k = i
            break
    selected_mean = means[k]
    selected_cov = covs[k]
    x = np.random.multivariate_normal(selected_mean, selected_cov)
    return x/np.sqrt(np.sum(x**2))