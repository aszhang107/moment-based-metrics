import numpy as np
from sympy.physics.quantum.cg import CG, cg_simp
from sympy import S, N
from scipy.io import loadmat, savemat

l_cap = 50 # Set this to whatever your L bandlimit is

P = l_cap // 5 # Can set this to whatever your P bandlimit is
CG_matrix = np.zeros((P + 1, l_cap + 1, l_cap + 1, 2 * l_cap + 1, 2 * l_cap + 1))


for l1 in range(l_cap + 1):
    print(l1)
    for l2 in range(l_cap + 1):
        for l3 in range(np.abs(l1 - l2), P + 1):
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    m1_index = m1 + l1
                    m2_index = m2 + l2
                    a = N(CG(l1,m1,l2,m2,l3,m1+m2).doit())
                    CG_matrix[l3, l1, l2, m1_index, m2_index] = a



savemat('clebsch-gordan.mat', {'data': CG_matrix})
