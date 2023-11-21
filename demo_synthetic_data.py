import time
import pickle
import multiprocessing
import numpy as np
import pyshtools as pysh
import prody
from fourier_transform_utils import get_inverse_fourier_transform
from generate_synthetic_molecule import generate_molecule_spectrum_from_pdb_id, center_atoms
from aspire.volume import Volume
from aspire.utils import Rotation
from aspire.noise.noise import WhiteNoiseAdder
from aspire.source.simulation import Simulation
from aspire.source.image import ArrayImageSource
from aspire.operators import RadialCTFFilter
from fle_2d_single import FLEBasis2D
from fast_cryo_pca import FastPCA 
import utils_cwf_fast_batch as utils
import finufft
from scipy.linalg import solve_triangular
from utils import compute_moments, read_pdb_ids_from_csv, rot_matrix_around_axis, generate_mixture_vonMises
from geomstats.geometry.hypersphere import Hypersphere

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
num_imgs = 50000 #number of images

r_image = np.fft.fftfreq(N, voxel_size)[N//2 - 1]
c_grid = np.linspace(0, r_image, N//2, endpoint=True)
phi_grid = np.linspace(0, 2*np.pi, phi_grid_size, endpoint=False)
r_size = len(c_grid)

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
        

#Generate viewing angle distribution
sphere = Hypersphere(dim=2)
mean1 = sphere.random_uniform()
mean2 = sphere.random_uniform()
mean3 = sphere.random_uniform()
cov1 = np.random.randn(3, 3)
cov2 = np.random.randn(3, 3)
cov3 = np.random.randn(3, 3)
cov1 = cov1.T @ cov1
cov2 = cov2.T @ cov2
cov3 = cov3.T @ cov3
pis = np.random.rand(3)
dir_samples = np.array([generate_mixture_vonMises(np.array([mean1, mean2, mean3]), np.array([cov1, cov2, cov3]), pis / np.sum(pis) ) for _ in range(num_imgs)])
angle_samples = np.random.uniform(0.0, 2*np.pi, num_imgs)
sampled_rotations = rot_matrix_around_axis(dir_samples, angle_samples)

eulers = Rotation(sampled_rotations).angles

#Generate volume from pdb
fixed_structure = '7VV3'
X = generate_molecule_spectrum_from_pdb_id(fixed_structure, voxel_size, phi_grid_size)
x = get_inverse_fourier_transform(X, voxel_size = voxel_size).real
vol = Volume(x)

#Create synthetic images
img_size = N
batch_size = 2 * num_imgs // 10
dtype = np.float64
eps = 6e-8
pixel_size = voxel_size # Pixel size of the images (in angstroms)
defocus_ct = 100 # the number of defocus groups
sn_ratio = 1/32

voltage = 200  # Voltage (in KV)
defocus_min = 1e4  # Minimum defocus value (in angstroms)
defocus_max = 3e4  # Maximum defocus value (in angstroms)
# Create filters. This is a list of CTFs. Each element corresponds to a UNIQUE CTF
h_ctf = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]
h_idx = utils.create_ordered_filter_idx(num_imgs, defocus_ct)


source_ctf_clean = Simulation(
    L=img_size,
    n=num_imgs,
    vols=vol,
    offsets=0.0,
    amplitudes=N,
    dtype=dtype,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    angles=eulers
)

# determine noise variance to create noisy images with certain SNR
noise_var = utils.get_noise_var_batch(source_ctf_clean, sn_ratio, batch_size)

# create simulation object for noisy images
source = Simulation(
    L=img_size,
    n=num_imgs,
    vols=vol,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    #offsets=0.0,
    amplitudes=N,
    dtype=dtype,
    noise_adder=WhiteNoiseAdder(noise_var),
    offsets = 1.5 * np.random.randn(num_imgs, 2),
    angles = eulers
)

# double the amount of simulated images by reflecting
original = source.images[:].asnumpy().copy() 
reflect = original[:, ::-1, :]
combine = np.vstack((original, reflect))
new_src = ArrayImageSource(combine)

# double the amount of clean images by reflecting
original = source_ctf_clean.projections[:].asnumpy().copy() 
reflect = original[:, ::-1, :]
combine = np.vstack((original, reflect))
new_src_clean = ArrayImageSource(combine)

# Fourier-Bessel expansion object
fle = FLEBasis2D(img_size, img_size, eps=eps)

# get clean sample mean and covariance
mean_clean = utils.get_clean_mean_batch(new_src_clean, fle, batch_size, True)
covar_clean = utils.get_clean_covar_batch(new_src_clean, fle, mean_clean, batch_size, dtype, True)

# options for covariance estimation
options = {
    "whiten": False,
    "single_pass": True, # whether estimate mean and covariance together (single pass over data), not separately
    "noise_var": noise_var, # noise variance
    "batch_size": batch_size,
    "dtype": dtype,
    "h_ctf": h_ctf,
    "h_idx": np.concatenate((h_idx, h_idx))
}

# create fast PCA object
fast_pca = FastPCA(new_src, fle, options)

# options for denoising
denoise_options = {
    "denoise_df_id": [0, 30, 60, 90], # denoise 0-th, 30-th, 60-th, 90-th defocus groups
    "denoise_df_num": [10, 15, 1, 240], # for each defocus group, respectively denoise the first 10, 15, 1, 100 images
                                        # 240 exceed the number of images (100) per defocus group, so only 100 images will be returned
    "return_denoise_error": True,
    "clean_src": new_src_clean,
    "store_images": True,
}

results = fast_pca.estimate_mean_covar(denoise_options=denoise_options)

mean_est = fast_pca.mean_est / N
covar_est = fast_pca.covar_est.mul(1 / (N**2))

err_mean = np.linalg.norm(mean_clean-mean_est)/np.linalg.norm(mean_clean)
_, err_covar = utils.compute_covar_err(covar_est, covar_clean)
print(f'error of mean estimation = {err_mean}')
print(f'error of covar estimation = {err_covar}')

#Convert m1 into appropriate basis and space
m1 = N * fle.evaluate(fle.to_eigen_order(mean_est))
vol = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(m1.reshape(N, N))))
m1_fourier = vol[N//2][N//2:] / np.sqrt(4 * np.pi)

#Convert m2 into appropriate basis and space
covar_ = covar_est @ np.eye(fle.ne)
m2_intermediate = np.zeros((fle.ne, N * N), dtype = np.float64)
m2_final = np.zeros((N * N, N * N), dtype = np.float64)
for i in range(fle.ne):
    m2_intermediate[i] = fle.evaluate(fle.to_eigen_order(covar_[i])).flatten()
m2_intermediate = m2_intermediate.T
for i in range(N * N):
    m2_final[i] = fle.evaluate(fle.to_eigen_order(m2_intermediate[i])).flatten()
m2_final = N**2 * m2_final.T.astype(np.complex128)
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

#Uncenter the moment
m2_uncentered = m2_reshaped_final + (np.expand_dims(m1_fourier, 1) @ np.expand_dims(m1_fourier, 0))

#Make m1 and m2 vector and make the weighted analogoues
experimental_data = np.concatenate((m1_fourier, m2_uncentered.flatten()))
experimental_data_weighted = np.concatenate((experimental_data[:N//2] * np.sqrt(c_grid), (experimental_data[N//2:].reshape(N, N//2, N//2) * \
                                                 (np.expand_dims(np.sqrt(c_grid), 1) @ np.expand_dims(np.sqrt(c_grid), 0))).flatten()))
experimental_data_weighted = experimental_data_weighted[experimental_data_weighted != 0]

#If we have the even QR matrices saved, use these
def obtain_LS_matrix_even_QR(pdb):
    with open('{}/ls_matrices/{}-even-qr.pickle'.format(save_path, pdb), 'rb') as handle:
        return pickle.load(handle)
#If we have the LS matrices saved, use these
def obtain_LS_matrix(pdb):
    with open('{}/ls_matrices/{}.pickle'.format(save_path, pdb), 'rb') as handle:
        return pickle.load(handle)
def metric_weighted(pdb):
    Q_mat, R_mat = obtain_LS_matrix_even_QR(pdb)
    lsmat = obtain_LS_matrix(pdb)[:, even_indices]
    B_hat = np.ones(lsmat.shape[1], dtype = np.complex128)
    B_hat[1:] = solve_triangular(R_mat, Q_mat.T.conj() @ (experimental_data_weighted - lsmat[:, 0]))
    return (pdb, np.linalg.norm(lsmat @ B_hat - experimental_data_weighted)**2)

#Rank database structures
starttime = time.time()

with multiprocessing.Pool(num_workers) as pool:
    distances_m2 = pool.map(metricWeighted, pdb_id_list)
print('Runtime: {} minutes'.format((time.time() - starttime)/60))
distances_m2_sorted = sorted(distances_m2, key=lambda tup: tup[1])