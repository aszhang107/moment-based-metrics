This is code accompanying the paper, [arxiv].

We provide 4 scripts: *demo_synthetic_data.py*, *demo_experimental_data.py*, *precompute_from_pdb.py*, *precompute_from_map.py*

A list of requirements are stored in *requirements.txt*, which can be installed via the command `pip install requirements.txt`.

To run the code, one must first generate least squares matrices using the appropriate precompute scripts. Here, we use *precompute_from_pdb.py* to generate the data needed for *demo_synthetic_data.py* and *precompute_from_map.py* to generate the data needed for *demo_experimental_data*. The directory to save/read data can be changed by editing the `save_path` variable in each script. The maps for *precompute_from_map.py* can also be downloaded at `https://www.ebi.ac.uk/emdb/EMD-NUM` where NUM's are listed in `emd_id_list` and resized (i.e. using https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize).

Once the precompute steps are done, the demos are ready to run. Note that *demo_experimental_data* requires the particle stack downloaded from https://www.ebi.ac.uk/empiar/EMPIAR-10081/. For ease of use, one can download the images to the *particle_stacks* folder in this directory.
