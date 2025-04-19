import numpy as np
from ete3 import Tree
import pandas as pd
import time

def load_data():
    data = np.loadtxt("/var/home/luka/proj/Papilonidae_dataset_v2/Papilionidae_aligned_new.txt", delimiter="\t").reshape((2240, 200))
    cats = pd.read_csv("/var/home/luka/proj/Papilonidae_dataset_v2/Papilonidae_metadata_new.txt", header=None)[0]

    tree = Tree("/var/home/luka/proj/Papilonidae_dataset_v2/papilionidae_tree.txt", format=1)
    order = tree.get_leaf_names()  # returns in-order leaves (same as pre-order when just looking at leaves)

    df = pd.DataFrame(data.reshape(2240, -1))
    df['category'] = pd.Categorical(cats, categories=cats.unique())
    means = df.groupby('category', observed=True).mean()
    df['order'] = pd.Categorical(df['category'], categories=order, ordered=True)
    means = means.reset_index()
    means['order'] = pd.Categorical(means['category'], categories=order, ordered=True)
    means = means.sort_values('order')

    x = means.drop(columns=['category', 'order']).values

    return x, tree

def design_matrix(n,m):
    des = np.zeros((n*m, m))
    i_indices = np.arange(n*m)
    j_indices = np.arange(m)
    # Use broadcasting to create a mask
    mask = (j_indices[:, None] * n <= i_indices) & (i_indices < (j_indices[:, None] + 1) * n)
    des[mask.T] = 1.0
    return des

def cov_matrix(tree):
    preorder = [node for node in tree.traverse("preorder")]
    m = len(preorder)  # number of nodes in tree (including internal)
    dists = np.zeros((m))

    for i in range(m):
        dists[i:i + len(preorder[i].get_descendants()) + 1] += preorder[i].dist

    leaves = tree.get_leaves()
    n = len(leaves)
    # lca_matrix = np.zeros((N, N), dtype=int)
    cov = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            ancestor = leaves[i].get_common_ancestor(leaves[j])
            lca_ij = preorder.index(ancestor)
            # lca_matrix[i, j] = lca_matrix[j, i] = lca_ij
            cov[i, j] = cov[j, i] = dists[lca_ij]

    return cov


# computes mle of phylogenetic mean (root shape) 'r', and covariance parameter 'R'
def mle_estimate(x,cov):
    n,d = x.shape
    v1 = np.ones(n)
    cov_inv = np.linalg.inv(cov)
    tmp = v1.T @ cov_inv
    mle_r = ((tmp @ v1) ** -1) * (tmp @ x)
    assert mle_r.shape == (d,)

    tmp = x - mle_r.T
    mle_R = (((n - 1) ** -1) * tmp.T) @ cov_inv @ tmp
    assert mle_R.shape == (d, d)

    return mle_r, mle_R

def ppca_init(tree,x):
    cov = cov_matrix(tree)
    mle_r, mle_R = mle_estimate(x,cov)
    evals, evecs = np.linalg.eigh(mle_R)
    x_centered = x - mle_r[None,:]
    return evals, evecs, x_centered, mle_r, mle_R

def ppca_recon(mle_r,x_cent,evecs,k=2):
    # X_mean = np.mean(X,axis=0)
    V_k = evecs[:, -k:]
    #print(x_cent.shape,V_k.shape)
    X_reduced = x_cent @ V_k
    #print(X_reduced.shape)
    X_reconstructed = X_reduced @ V_k.T + mle_r[None, :]
    #print(X_reconstructed.shape)
    return X_reconstructed


# t0: variable containing previous time.time()
global t0,times,ti
p1 = 9      # constants for padding print statements
p2 = 4
p3 = p1+p2

ts = 48+4


def time_init():
    global t0,ti,times
    t0 = time.time()
    ti = 0
    times = np.empty((ts,), dtype=float)
    print(f'{"(time [ms])":<{p3}}{"(comment)"}\n')

def logtime(comment=""):
    global t0,ti,times
    t1 = time.time()
    res = (t1-t0)*1000
    times[ti] = res
    print(f'{res:>{p1}.6f}{"":<{p2}}{comment}')
    t0 = t1
    ti += 1

def main():
    time_init()
    X, ptree = load_data()
    logtime("load data")

    N, D = X.shape
    Des = design_matrix(N, D)
    logtime("design matrix")

    Cov = cov_matrix(ptree)
    logtime("covariance matrix")

    Evals, Evecs, X_cent, Mle_r, Mle_R = ppca_init(ptree,X)
    logtime("ppca init")

    print(f'\n{"":<{p3}}{"ppca reconstruction"}')
    for K in range(1,N+1):
        krec = ppca_recon(Mle_r,X_cent,Evecs,K)
        np.savetxt(f'out/{K}-recon.txt', krec, delimiter='\t')
        logtime(f"k={K}")

main()
