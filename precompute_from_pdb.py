import time
import pickle
import multiprocessing
import numpy as np
import pyshtools as pysh
import prody
import scipy.sparse
from scipy.io import loadmat
from generate_synthetic_molecule import generate_potential_at_freqs_from_atoms, center_atoms
from utils import compute_moments, read_pdb_ids_from_csv, Molecule

pdb_file = 'pdb_id_list.txt' #file containing pdb ids
pdb_id_list = read_pdb_ids_from_csv(pdb_file) 

l_cap = 25 #L - bandlimit parameter
N = 64 #grid size
p_cap = 6 #P - bandlimit parameter
voxel_size = 200 / N #voxel size in angstroms
phi_grid_size = N #grid of angles
Nsph = 300 #bandlimit for spherical harmonics expansion
save_path = '/scratch/gpfs/az8940' #where to save to
prody.pathPDBFolder('{}/pdbs'.format(save_path)) #directory with pdbs
num_workers = 16 #workers for parallel processing

r_image = np.fft.fftfreq(N, voxel_size)[N//2 - 1]
c_grid = np.linspace(0, r_image, N//2, endpoint=True)
phi_grid = np.linspace(0, 2*np.pi, phi_grid_size, endpoint=False)
r_size = len(c_grid)

CG_matrix = loadmat('clebsch-gordan.mat')['data']
N_ = np.array(loadmat('N_matrix.mat')['data'])

def get_moments(pdb):
    try:
        atom_group = center_atoms(prody.parsePDB(pdb))
        M1, M2, As = compute_moments(atom_group, Nsph, phi_grid, c_grid, voxel_size)
        molecule = Molecule(pdb, atom_group.numAtoms(), M1, M2)
    except Exception as e:
        print(e)
        print('ERROR with pdb {}'.format(pdb))
        molecule = Molecule(pdb, 0, 0, 0)
        As = []
    return molecule, As

index_dict = {}
m_list = []
even_indices = []

i = 0
for l_prime in range(0, p_cap + 1):
    for m in range(-l_prime, l_prime + 1):
        index_dict[(l_prime, m)] = i
        m_list.append((l_prime, m))
        if l_prime % 2 == 0:
            even_indices.append(i)
        i += 1

def scriptB_LS(l1, l2, n):
    LS_mat = np.zeros(((2 * l1 + 1) * (2 * l2 + 1), len(index_dict)), dtype=np.complex128)
    row_idx = 0
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for p in range(max(np.abs(l1 - l2), np.abs(m1 + m2)), min(l1 + l2, p_cap) + 1):
                col_idx = index_dict[(p, -m1 - m2)]
                LS_mat[row_idx, col_idx] += (-1)**(np.abs(m1 + m2)) / (2 * p + 1) * CG_matrix[p, l1, l2, m1 + l1, m2 + l2] * CG_matrix[p, l1, l2, n + l1, -n + l2] 
            row_idx += 1
    return LS_mat    

def generate_B_matrices(l_cap):
    B_mat_dict = {}
    for n in range(-l_cap, l_cap + 1):
        for l1 in range(np.abs(n), l_cap + 1, 2):
            for l2 in range(np.abs(n), l_cap + 1, 2):
                B_mat_dict['n={},l1={},l2={}'.format(n, l1, l2)] = scipy.sparse.csr_matrix(scriptB_LS(l1, l2, n))
    return B_mat_dict

def precompute_step(pdb):
    t1 = time.time()
    mol, A = get_moments(pdb)
    if len(A) == 0:
        print("Moment cannot be calculated with error in pdb {}".format(pdb))
        #Record invalid values
        '''
        with open('{}/invalid_pdbs/{}.pickle'.format(save_path, pdb), 'wb') as handle:
            pickle.dump([], handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''
        return
    t2 = time.time()
    print('Coefficients generated, time = {}'.format(t2 - t1))
    A_list = []
    for i in range(l_cap + 1):
        A_list.append(np.concatenate((A[i][:, -149 + i -1 : -149-1 : -1].T, A[i][:, :i+1].T)).T)
    LS_matrix = np.zeros((r_size**2 * phi_grid_size + r_size, len(index_dict)), dtype = np.complex128)
    for r in range(r_size):
        for l in range(p_cap + 1):
            for m in range(-l, l + 1):
                col_idx = index_dict[(l, -m)]
                # += and = are the same here since for each entry you will hit each entry of B_{l,m} at most once
                LS_matrix[r, col_idx] = A_list[l][r, m + l] * N_[l, l] * 1 / (2 * l + 1) * (-1)**np.abs(m)
    nonscaledLS = {}
    for n in range(-l_cap, l_cap + 1):
        for l1 in range(np.abs(n), l_cap + 1, 2):
            for l2 in range(np.abs(n), l_cap + 1, 2):
                nonscaledLS[(n, l1, l2)] = N_[n + l1, l1] * N_[-n + l2, l2] * np.kron(A_list[l1], A_list[l2]) @ B_ls_matrices['n={},l1={},l2={}'.format(n, l1, l2)]
    t3 = time.time()
    print('Precompute done, time = {}'.format(t3 - t2))
    for phi_idx in range(phi_grid_size):
        phi_diff = phi_grid[phi_idx]
        for n in range(-l_cap, l_cap + 1):
            for l1 in range(np.abs(n), l_cap + 1, 2):
                for l2 in range(np.abs(n), l_cap + 1, 2):
                    LS_matrix[r_size**2 * phi_idx + r_size:r_size**2 * (phi_idx + 1) + r_size] += np.exp(1j * n * phi_diff) * nonscaledLS[(n, l1, l2)]
    t4 = time.time()
    print('Moment generation done, time = {}'.format(t4 - t3))
    print('QR factorization done, time = {}'.format(time.time() - t4))
    LS_matrix[:N//2] = (LS_matrix[:N//2].T * np.sqrt(c_grid)).T
    LS_matrix[N//2:] = (LS_matrix[N//2:].T.reshape(-1, N, N//2, N//2) * (np.expand_dims(np.sqrt(c_grid), 1) @ np.expand_dims(np.sqrt(c_grid), 0))).reshape(-1, N * N//2 * N//2).T #weight rows of LS matrix
    LS_matrix = LS_matrix[~np.all(LS_matrix == 0, axis=1)] #remove 0 rows from weighting
    
    #Save the LS matrix and its (even-indices) QR factorization and its first and second moment
    '''
    with open('{}/ls_matrices/{}.pickle'.format(save_path, pdb_), 'wb') as handle:
        pickle.dump(LS_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}/ls_matrices/{}-qr.pickle'.format(save_path, pdb_), 'wb') as handle:
        q, r = np.linalg.qr(LS_matrix[:, 1:])
        pickle.dump((q, r), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}/ls_matrices/{}-even-qr.pickle'.format(save_path, pdb_), 'wb') as handle:
        q, r = np.linalg.qr(LS_matrix[:, even_indices][:, 1:])
        pickle.dump((q, r), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}/uniform_moments/{}.pickle'.format(save_path, pdb_), 'wb') as handle:
        pickle.dump(mol, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''

B_ls_matrices = generate_B_matrices(l_cap)

#Run the precompute step for every pdb id       
starttime = time.time()

with multiprocessing.Pool(num_workers) as pool:
    pdb_moments = pool.map(precompute_step, pdb_id_list)

print('Runtime: {} minutes'.format((time.time() - starttime)/60))