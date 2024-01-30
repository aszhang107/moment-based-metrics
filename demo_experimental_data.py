import math
import time
import pickle
import multiprocessing
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import pyshtools as pysh
from scipy.special import sph_harm
from aspire.volume import Volume
from aspire.utils.coor_trans import grid_3d, grid_2d
from fle_2d_single import FLEBasis2D
from fast_cryo_pca import FastPCA 
import utils_cwf_fast_batch as utils
import finufft
from scipy.linalg import solve_triangular
from aspire.source.relion import RelionSource

vol_id_list = ['000', '001', '002', '003', '004']

l_cap = 25 #L - bandlimit parameter
N = 64 #grid size
p_cap = 6 #P - bandlimit parameter
voxel_size = 1.3 * 4 #voxel size in angstroms
phi_grid_size = N #grid of angles
Nsph = 300 #bandlimit for spherical harmonics expansion
save_path = '/scratch/gpfs/az8940/empiar10076-analysis/' #where to save to
num_workers = 16 #workers for parallel processing

r_image = np.fft.fftfreq(N, voxel_size)[N//2 - 1]
c_grid = np.linspace(0, r_image, N//2, endpoint=True)
phi_grid = np.linspace(0, 2*np.pi, phi_grid_size, endpoint=False)
r_size = len(c_grid)

index_dict = {}
m_list = []

i = 0
for l_prime in range(0, p_cap + 1):
    for m in range(-l_prime, l_prime + 1):
        index_dict[(l_prime, m)] = i
        m_list.append((l_prime, m))
        i += 1

def get_experimental_data_weighted(vol_id):
    logger = logging.getLogger(__name__)
    # Set input path and files and initialize other parameters
    DATA_FOLDER = 'EMPIAR10076_data'
    STARFILE_IN = 'EMPIAR10076_data/Parameters-major-{}.star'.format(vol_id)

    MAX_ROWS = None
    eps = 6e-8
    batch_size = 5000
    MAX_RESOLUTION = N
    PIXEL_SIZE = 1.31
    dtype = np.float32
    # Create a source object for 2D images
    print(f'Read in images from {STARFILE_IN} and preprocess the images.')
    source = RelionSource(
        STARFILE_IN,
        DATA_FOLDER,
        pixel_size=PIXEL_SIZE,
        max_rows=MAX_ROWS,
        n_workers=60
    )

    #print(len(source.unique_filters))
    print(f'Set the resolution to {MAX_RESOLUTION} X {MAX_RESOLUTION}')
    if MAX_RESOLUTION < source.L:
        source = source.downsample(MAX_RESOLUTION)

    shifts = -poses[1][key_dict[vol_id]] * N
    fle = FLEBasis2D(MAX_RESOLUTION, MAX_RESOLUTION, eps=eps, dtype=dtype)

    options = {
        "whiten": True,
        "noise_psd": None,
        "radius": 0.8,
        "single_pass": True,
        "flip_sign": False,
        "single_whiten_filter": False,
        "batch_size": batch_size,
        "dtype": dtype,
        "correct_contrast": True,
        "subtract_background": True,
        "shifts": shifts
    }

    fast_pca = FastPCA(source, fle, options)

    fast_pca.estimate_mean_covar()

    mean_est = fast_pca.mean_est
    covar_est = fast_pca.covar_est

    m1 = fle.evaluate(fle.to_eigen_order(mean_est))
    vol = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(m1.reshape(N, N))))
    m1_fourier = vol[N//2][N//2:] * N_[0, 0]

    covar_ = covar_est @ np.eye(fle.ne)
    m2_intermediate = np.zeros((fle.ne, N * N), dtype = np.float64)
    m2_final = np.zeros((N * N, N * N), dtype = np.float64)
    for i in range(fle.ne):
        m2_intermediate[i] = fle.evaluate(fle.to_eigen_order(covar_[i])).flatten()
    m2_intermediate = m2_intermediate.T
    for i in range(N * N):
        m2_final[i] = fle.evaluate(fle.to_eigen_order(m2_intermediate[i])).flatten()
    m2_final = m2_final.T.astype(np.complex128)
    m2_intermediate_fourier = np.zeros((N * (N // 2), N * N), dtype = np.complex128)
    m2_final_fourier = np.zeros((N * (N // 2), N * (N // 2)), dtype = np.complex128)
    rs, thetas = np.meshgrid(c_grid, phi_grid)
    rs = rs.flatten()
    thetas = thetas.flatten()
    x_grid = (N - 2) / N * rs * np.cos(thetas) * np.pi / r_image
    y_grid = (N - 2) / N * rs * np.sin(thetas) * np.pi / r_image
    x_grid = x_grid.astype(np.float64).flatten()
    y_grid = y_grid.astype(np.float64).flatten()
    for i in range(N * N):
        m2_intermediate_fourier[:, i] = finufft.nufft2d2(x_grid, -y_grid, m2_final[:, i].reshape(N, N).T)
    m2_intermediate_fourier = m2_intermediate_fourier.conj().T
    for i in range(N * (N // 2)):
        m2_final_fourier[:, i] = finufft.nufft2d2(x_grid, -y_grid, m2_intermediate_fourier[:, i].reshape(N, N).T)
    m2_final_fourier = m2_final_fourier.conj().T

    m2_reshaped = np.zeros((N, N//2, N//2), dtype = np.complex128)
    for i in range(N):
        m2_reshaped[i] = m2_final_fourier[:N//2, N//2 * i: N//2 * (i + 1)]

    m2_reshaped_final = np.zeros(m2_reshaped.shape, dtype = np.complex128)
    m2_reshaped_final[:N//2] = m2_reshaped[N//2:]
    m2_reshaped_final[N//2:] = m2_reshaped[:N//2]
    m2_reshaped_final /= 4 * np.pi

    m2_uncentered = m2_reshaped_final + (np.expand_dims(m1_fourier, 1) @ np.expand_dims(m1_fourier, 0))
    experimental_data = np.concatenate((m1_fourier, m2_uncentered.flatten()))
    experimental_data_weighted = np.concatenate((experimental_data[:N//2] * np.sqrt(c_grid), (experimental_data[N//2:].reshape(N, N//2, N//2) * \
                                                     (np.expand_dims(np.sqrt(c_grid), 1) @ np.expand_dims(np.sqrt(c_grid), 0))).flatten()))
    experimental_data_weighted = experimental_data_weighted[experimental_data_weighted != 0]
    #with open('{}/experimental_datas_major_{}'.format(save_path, vol_id), 'wb') as f:
    #    pickle.dump((experimental_data, experimental_data_weighted), f)
    return experimental_data_weighted, experimental_data #m2_uncentered

#Retrieve LS matrices if they are stored
def obtain_LS_matrix(vol_id):
    with open('{}/ls_matrices_from_map/{}.pickle'.format(save_path, vol_id), 'rb') as handle:
        return pickle.load(handle)

def obtain_unweighted_LS_matrix(vol_id):
    with open('{}/unweighted_ls_matrices_from_map/{}.pickle'.format(save_path, vol_id), 'rb') as handle:
        return pickle.load(handle)
    
def metric_weighted(vol_id):
    LS_matrix = obtain_unweighted_LS_matrix(vol_id)
    
    #First find the scale
    uniformB = np.zeros(LS_matrix_u.shape[1], dtype = np.complex128)
    uniformB[0] = 1
    m2_B = (LS_matrix_u @ uniformB)[N//2:].reshape(N, N//2, N//2)
    b = np.diag(experimental_data[32:].reshape(64, 32, 32)[0])
    a = np.diag(m2_B[0])
    factor = np.sqrt(a.T @ b / (a.T @ a)).real
    LS_matrix[:N//2] *= factor
    LS_matrix[N//2:] *= factor**2
    
    # Weight the matrix
    LS_matrix[:N//2] = (LS_matrix[:N//2].T * np.sqrt(c_grid)).T
    LS_matrix[N//2:] = (LS_matrix[N//2:].T.reshape(-1, N, N//2, N//2) * (np.expand_dims(np.sqrt(c_grid), 1) @ np.expand_dims(np.sqrt(c_grid), 0))).reshape(-1, N * N//2 * N//2).T
    LS_matrix = LS_matrix[~np.all(LS_matrix == 0, axis=1)]
    
    #Once scaled, now solve the equation
    Q_mat, R_mat = np.linalg.qr(LS_matrix[:, 1:])
    B_hat = np.ones(LS_matrix.shape[1], dtype = np.complex128)
    B_hat[1:] = solve_triangular(R_mat, Q_mat.T.conj() @ (experimental_data_weighted - LS_matrix[:, 0]))
    return (vol_id, np.linalg.norm(LS_matrix @ B_hat - experimental_data_weighted)**2)

#Rank database structures
starttime = time.time()

with multiprocessing.Pool(num_workers) as pool:
    distances_m2 = pool.map(metricWeighted, vol_id_list)
print('Runtime: {} minutes'.format((time.time() - starttime)/60))
distances_m2_sorted = sorted(distances_m2, key=lambda tup: tup[1])