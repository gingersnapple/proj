import numpy as np
from ete3 import Tree
import pandas as pd
import time


# Rapport: husk rød tråd, antagelser, læserens forudsætninger (sig det explicit)
# TODO: artificielle træer, se udvikling af køretider som f. af datasæt størrelse
# TODO: generer data, træer i N størrelse
# tal om evt. data påvirker eigendecomposition.. sammenlign med original data

def convert_to_slice(df_subgroup):
    x_coords = df_subgroup[df_subgroup.iloc[:, 1] == "x-coordinates"].iloc[:, 2:].values
    y_coords = -df_subgroup[df_subgroup.iloc[:, 1] == "y-coordinates"].iloc[:, 2:].values
    return np.column_stack((x_coords.ravel(), y_coords.ravel()))


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

def design_matrix(n, m):
    des = np.zeros((n * m, m))

    for i in range(n * m):
        for j in range(m):
            if (j - 1) * n < i >= j * n:
                des[i, j] = 1.

    return des


# np.set_printoptions(threshold=sys.maxsize)
def cov_matrix(preorder, n):
    m = len(preorder)
    dists = np.zeros(m)
    lcas = np.zeros((m, m), dtype=int)
    leaf_inds = np.empty(n, dtype=int)
    k = 0

    sizes = np.zeros(m, dtype=int)

    def get_size(ind):
        sz = sizes[i]
        if sz == 0:
            sz = len(preorder[ind].get_descendants()) + 1
            sizes[i] = sz
        return sz

    for i in range(m):
        s = get_size(i)
        dists[i:i + s] += preorder[i].dist
        if s >= 2:
            l = i + 1
            r = l + get_size(l)
            end = r + get_size(r)
            # ugly, use mask instead?
            lcas[i:r, r:end] = lcas[r:end, i:r] = lcas[i, i:r] = lcas[i:r, i] = i
        else:
            leaf_inds[k] = i
            k += 1
        lcas[i, i] = i

    leaf_lcds = lcas[leaf_inds][:, leaf_inds]
    cov = dists[leaf_lcds]

    return cov


# computes mle of phylogenetic mean (root shape) 'r', and covariance parameter 'R'
def mle_estimate(x, cov):
    n, d = x.shape
    v1 = np.ones(n)
    cov_inv = np.linalg.inv(cov)

    tmp1 = v1.T @ cov_inv
    mle_r = ((tmp1 @ v1) ** -1) * (tmp1 @ x)
    x_centered = x - mle_r[None, :]

    tmp2 = x - mle_r.T
    mle_R = (((n - 1) ** -1) * tmp2.T) @ cov_inv @ tmp2

    return mle_r, mle_R, x_centered


def ppca_init(x, preorder, leaves):
    n, d = x.shape
    cov = cov_matrix(preorder, n)
    logtime("covariance matrix")

    mle_r, mle_R, x_centered = mle_estimate(x, cov)
    logtime("mle calculations")

    evals, evecs = np.linalg.eigh(mle_R)
    logtime("eigen decomposition")

    return evals, evecs, x_centered, mle_r, mle_R


def ppca_recon(mle_r, x_cent, evecs, k=2):
    # X_mean = np.mean(X,axis=0)
    V_k = evecs[:, -k:]
    # print(x_cent.shape,V_k.shape)
    X_reduced = x_cent @ V_k
    # print(X_reduced.shape)
    X_reconstructed = X_reduced @ V_k.T + mle_r[None, :]
    # print(X_reconstructed.shape)
    return X_reconstructed


# t0: variable containing previous time.time()
global t0, times, ti
p1 = 9  # constants for padding print statements
p2 = 4
p3 = p1 + p2

ts = 128


def time_init():
    global t0, ti, times
    t0 = time.time()
    ti = 0
    times = np.empty((ts,), dtype=float)
    print(f'{"(time [ms])":<{p3}}{"(comment)"}\n')


def logtime(comment=""):
    global t0, ti, times
    t1 = time.time()
    res = (t1 - t0) * 1000
    times[ti] = res
    print(f'{res:>{p1}.6f}{"":<{p2}}{comment}')
    t0 = t1
    ti += 1


def recons(mle_r, x_cent, evecs):
    N, D = x_cent.shape
    print(f'\n{"":<{p3}}{"ppca reconstruction"}')
    for K in range(1, N + 1):
        krec = ppca_recon(mle_r, x_cent, evecs, K)
        logtime(f"k={K}")
        np.savetxt(f'out/{K}-recon.txt', krec, delimiter='\t')
        logtime("printout")


def main():
    time_init()
    X, ptree, preorder, leaves = load_data()

    Evals, Evecs, X_cent, Mle_r, Mle_R = ppca_init(X, preorder, leaves)

    outs = [Evals, Evecs, X_cent, Mle_r, Mle_R]

    for i in range(5):
        np.savetxt(f'out/ppca-init-{i}.txt', outs[i], delimiter='\t')

    logtime("printout")

    recons(Mle_r, X_cent, Evecs)


main()
