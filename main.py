import numpy as np
import sys
from ete3 import Tree
import pandas as pd
import time


def load_data():
    data = np.loadtxt("/var/home/luka/proj/Papilonidae_dataset_v2/Papilionidae_aligned_new.txt", delimiter="\t").reshape((2240, 200))
    logtime("load marks")

    cats = pd.read_csv("/var/home/luka/proj/Papilonidae_dataset_v2/Papilonidae_metadata_new.txt", header=None)[0]
    logtime("load metadata")

    tree = Tree("/var/home/luka/proj/Papilonidae_dataset_v2/papilionidae_tree.txt", format=3)
    leaf_names = tree.get_leaf_names()  # returns in-order leaves (same as pre-order when just looking at leaves)
    preorder = [n for n in tree.traverse('preorder')]
    leaves = [n for n in preorder if n.is_leaf()]
    logtime("load tree")

    df = pd.DataFrame(data.reshape(2240, -1))
    df['category'] = pd.Categorical(cats, categories=cats.unique())
    means = df.groupby('category', observed=True).mean()
    df['order'] = pd.Categorical(df['category'], categories=leaf_names, ordered=True)
    means = means.reset_index()
    means['order'] = pd.Categorical(means['category'], categories=leaf_names, ordered=True)
    means = means.sort_values('order')
    x = means.drop(columns=['category', 'order']).values
    logtime("rework data")

    return x, tree, preorder, leaves

def design_matrix(n,m):
    des = np.zeros((n*m, m))
    i_indices = np.arange(n*m)
    j_indices = np.arange(m)
    # Use broadcasting to create a mask
    mask = (j_indices[:, None] * n <= i_indices) & (i_indices < (j_indices[:, None] + 1) * n)
    des[mask.T] = 1.0
    return des


#np.set_printoptions(threshold=sys.maxsize)
def cov_matrix(preorder, leaves):
    m = len(preorder)  # number of nodes in tree (including internal)
    dists = np.zeros(m)

    n = len(leaves)
    # lca_matrix = np.zeros((N, N), dtype=int)
    cov = np.zeros((n, n))

    lcds = np.zeros((m,m),dtype=int)
    leaf_indices = np.empty(n, dtype=int)
    k = 0
    for i in range(m):
        node = preorder[i]
        if node.is_leaf():
            leaf_indices[k] = i
            k += 1

        branch_size = len(node.get_descendants())+1

        dists[i:i+branch_size] += node.dist

        if branch_size >= 2:
            l = i+1
            r = l+len(preorder[l].get_descendants())+1
            end = r+len(preorder[r].get_descendants())+1
            lcds[l:r,r:end+1] = i
            lcds[r:end+1,l:r] = i

        lcds[i,i] = i


    leaf_lcds = lcds[leaf_indices][:,leaf_indices]
    print(leaf_lcds[2,2])
            # left_node = preorder[l]
            # right_node = preorder[r]
            # end_node = preorder[end-1]
            #print(left_node.get_common_ancestor(right_node) == node)
            #print(left_node.get_common_ancestor(end_node) == node)
            #print(preorder[i+left_size-1].is_leaf())
            #child2 = preorder[i + 1]
            #child2 = node.children[1]

            #


    logtime("cov 1")



    # TODO: optimize this more? maybe implement own get_common_ancestor function

    k = 0

    lcds_old = np.zeros((n,n),dtype=int)

    for i in range(n):
        for j in range(i, n):
            ancestor = leaves[i].get_common_ancestor(leaves[j])
            lca_ij = preorder.index(ancestor)
            # lca_matrix[i, j] = lca_matrix[j, i] = lca_ij
            lcds_old[i,j] = lcds_old[j,i] = lca_ij
            cov[i, j] = cov[j, i] = dists[lca_ij]

            if leaf_lcds[i, j] != lca_ij:
                print("A",i, j, leaf_lcds[i, j], lca_ij)
            if leaf_lcds[j, i] != lca_ij:
                print("B",i, j, leaf_lcds[i, j], lca_ij)

    logtime("cov 2")
    #print(lcds)
    #print(lcds[0,0],lcds_old[0,0])
    #print(leaf_lcds == lcds_old)
    return cov


# computes mle of phylogenetic mean (root shape) 'r', and covariance parameter 'R'
def mle_estimate(x,cov):
    n,d = x.shape
    v1 = np.ones(n)
    cov_inv = np.linalg.inv(cov)
    logtime("inv")

    tmp = v1.T @ cov_inv
    mle_r = ((tmp @ v1) ** -1) * (tmp @ x)
    assert mle_r.shape == (d,)
    x_centered = x - mle_r[None, :]
    logtime("mle r")


    tmp = x - mle_r.T
    mle_R = (((n - 1) ** -1) * tmp.T) @ cov_inv @ tmp
    assert mle_R.shape == (d, d)
    logtime("mle R")

    return mle_r, mle_R, x_centered

def ppca_init(x,preorder,leaves):
    cov = cov_matrix(preorder,leaves)
    #logtime("covariance matrix")

    mle_r, mle_R,x_centered = mle_estimate(x,cov)

    evals, evecs = np.linalg.eigh(mle_R)
    logtime("eigen decomposition")

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

ts = 64


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

def recons(mle_r,x_cent,evecs):
    N, D = x_cent.shape
    print(f'\n{"":<{p3}}{"ppca reconstruction"}')
    for K in range(1,N+1):
        krec = ppca_recon(mle_r,x_cent,evecs,K)
        np.savetxt(f'out/{K}-recon.txt', krec, delimiter='\t')
        logtime(f"k={K}")

def main():
    time_init()
    X, ptree, preorder, leaves = load_data()

    Evals, Evecs, X_cent, Mle_r, Mle_R = ppca_init(X,preorder,leaves)

    outs = [Evals, Evecs, X_cent, Mle_r, Mle_R]

    for i in range(5):
        np.savetxt(f'out/ppca-init-{i}.txt', outs[i], delimiter='\t')

    logtime("printout")


main()
