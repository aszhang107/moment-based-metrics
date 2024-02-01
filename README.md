This is code accompanying the paper, https://arxiv.org/abs/2401.15183.

We provide 4 scripts: *demo_synthetic_data.py*, *demo_experimental_data.py*, *precompute_from_pdb.py*, *precompute_from_map.py*

A list of requirements are stored in *requirements.txt*, which can be installed via the command `pip install -r requirements.txt`.

To run the code, one must first generate Clebsch-Gordan coefficients by running *utils/generate_CG_coefficients.py*. Then one can createleast squares matrices using the appropriate precompute scripts. Here, we use *precompute_from_pdb.py* to generate the data needed for *demo_synthetic_data.py* and *precompute_from_map.py* to generate the data needed for *demo_experimental_data*. The directory to save/read data can be changed by editing the `save_path` variable in each script.

Once the precompute steps are done, the demos are ready to run. Note that *demo_experimental_data* requires the particle stack downloaded from https://www.ebi.ac.uk/empiar/EMPIAR-10076/. For ease of use, one can download the images to the *EMPIAR-10076/* folder in this directory.

The code in *utils/fast_cryo_pca.py*, *utils/utils_cwf_fast_batch.py*, and *utils/fle_2d_single.py* are cloned from https://github.com/yunpeng-shi/fast-cryoEM-PCA/tree/main with minor changes to include more parameters.
