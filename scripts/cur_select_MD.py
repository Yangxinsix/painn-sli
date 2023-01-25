import numpy as np
from multiprocessing import Pool
from xinyang_toolbox import write_runner_input
from CUR import *
import os
import subprocess
from ase.io import read, write

# initialize parameters
processors = os.cpu_count()
mat_num = 800

images = read('MD.traj', index=':')
write_runner_input(images, convert_unit=False, check_force=True)
symfunc = subprocess.run('mpirun -np {} nnp-scaling 500'.format(processors), shell=True)
rm_file = subprocess.run('rm sf* nnp-scaling* scaling* neighbor* input.data', shell=True)

# read symmetry function values
element_sfs = read_symfunc()
for key in element_sfs.keys():
    np.random.shuffle(element_sfs[key])
rm_symfunc = subprocess.run('rm function.data input.nn', shell=True)

# calculate W_mat
atoms_l2_norm = []
for sfs in element_sfs.values():
    split_sfs = div_mat(sfs[:, 3:], mat_num)
    with Pool(processors) as pool:
        W_mat, C_arr = [], []
        for W, C in pool.map(CUR_decomposition_row, split_sfs):
            W_mat.append(W)
            C_arr.append(C)
    W_mat = np.concatenate(W_mat, axis=1)
    atoms_l2_norm.append(np.concatenate((sfs[:, :3], linalg.norm(W_mat, axis=0).reshape(-1,1)), axis=1))

# get image l2 norms
atoms_l2_norm = np.concatenate(atoms_l2_norm, axis=0)
np.save('atomic_norms', atoms_l2_norm)

image_indexs = np.unique(atoms_l2_norm[:,1])
images_l2_norm = np.empty((len(image_indexs), 2))
for i, image_index in enumerate(image_indexs):
    indexs = np.where(atoms_l2_norm[:,1]==image_index)
    images_l2_norm[i,0] = int(image_index)
    images_l2_norm[i,1] = np.mean(atoms_l2_norm[indexs, 3])

np.save('image_norms', images_l2_norm)
