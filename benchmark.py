import numpy as np
import time
import ngesh
import matplotlib.pyplot as plt
from ete3 import Tree as eteTree


from fast import cov_matrix, mle_estimate, ppca_recon
from preload import load_generated_data, generate_data, sample_size, batch_size, ns

def benchmark(w:int):
    print("Running benchmark...")
    times = np.empty((4, batch_size, sample_size))
    for n in range(sample_size):
        N = ns[n]
        print("N:", N)
        for i in range(batch_size):
            (x, preorder, leaves, tree) = load_generated_data(n)
            t0 = time.time()
            cov, m = cov_matrix(preorder, N)
            t1 = time.time()
            r, R, x_cent = mle_estimate(x, cov)  # singular matrix error
            t2 = time.time()
            _, evecs = np.linalg.eigh(R)
            t3 = time.time()
            ppca_recon(r, x_cent, evecs, k=2)
            t4 = time.time()

            ts = [t0, t1, t2, t3, t4]
            for j in range(4):
                times[j, i, n] = ts[j + 1] - ts[j]
            print(times[:, i, n]*1000,'\n')
    np.save(f'times_{w:0>2}', times)


def load_times(w:int):
    times = np.load(f'times_{w:0>2}.npy')
    avg = np.mean(times*1000,axis=1)    # avg times (ms)
    print(f'{"N":<6}{"cov":<16}{"mle":<16}{"eig":<16}{"recon":<16}')
    for n in range(sample_size):
        N = ns[n]
        print(f'{N:<6}{avg[0,n]:<16.4f}{avg[1,n]:<16.4f}{avg[2,n]:<16.4f}{avg[3,n]:<16.4f}')

    fnames = ["cov", "mle", "eig", "recon"]
    for i in range(4):
        plt.plot(ns,avg[i],'b.')
        plt.title(fnames[i])
        plt.savefig(f'timeplot_{fnames[i]}.png')
        plt.show()

#generate_data()
#benchmark(1)
load_times(1)