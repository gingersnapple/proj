import numpy as np
import time
from ete3 import Tree

from fast import cov_matrix, mle_estimate, ppca_recon
from preload import load_original_data, load_generated_data, generate_sample

def compare():
    x,preorder,leaves,tree = load_original_data()
    N, D = x.shape
    K = 100
    times = np.empty((2,4,K))

    for i in range(K):
        g_x,g_preorder,g_leaves,g_tree = generate_sample(n=N, d=D, seed=None)
        data_set = ((x,preorder,leaves,tree), (g_x,g_preorder,g_leaves,g_tree))
        print(i)
        for s in [0,1]:
            print(f' {s}')
            X,Preorder,Leaves,Ptree = data_set[s]
            #print (Ptree)
            t0 = time.time()
            Cov,M = cov_matrix(Preorder,N)
            t1 = time.time()
            Mle_r,Mle_R,X_cent = mle_estimate(X,Cov)    # singular matrix error
            t2 = time.time()
            _,Evecs = np.linalg.eigh(Mle_R)
            t3 = time.time()
            ppca_recon(Mle_r,X_cent,Evecs,k=2)
            t4 = time.time()

            ts = [t0,t1,t2,t3,t4]
            for j in range(4):
                times[s, j, i] = ts[j+1]-ts[j]


    np.save('times',times)

compare()