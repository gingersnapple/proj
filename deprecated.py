import numpy as np
import ngesh
from dendropy import Tree as denTree
from dendropy.calculate.treemeasure import colless_tree_imbalance


def compare(w:int):
    x,preorder,leaves,tree = load_original_data()
    N, D = x.shape
    K = 100
    times = np.empty((2,4,K))

    for i in range(K):
        g_x,g_preorder,g_leaves,g_tree = generate_sample(seed=None)
        data_set = ((x,preorder,leaves,tree), (g_x,g_preorder,g_leaves,g_tree))
        print(i,len(g_preorder))
        if i> (K//2):
            ss = [1,0]
        else:
            ss = [0,1]
        for s in ss:
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


    np.save(f'times_{w:0>2}',times)

def load_times(w:int):
    times = np.load(f'times_{w:0>2}.npy')
    avg = np.mean(times*1000,axis=2)    # avg times (ms)

    fnames = ["cov","mle","eig","recon"]
    print(f'{"op":<16}{"org":<16}{"gen":<16}')
    for i in range(4):
        print(f'{fnames[i]:<16}{avg[0,i]:<16.4f}{avg[1,i]:<16.4f}')



def get_depth(root,node):
    d = 0
    while not node == root:
        node = node.up
        d += 1
    return d


def analyze_tree(tree):
    nodes = [n for n in tree.traverse('preorder')]
    num_nodes = len(nodes)
    leaves = tree.get_leaves()
    num_leaves = len(leaves)
    max_depth = max([get_depth(tree,leaf) for leaf in leaves])

    nw_str = tree.write(format=1)
    den_tree = denTree.get(data=nw_str,schema='newick')
    imb = colless_tree_imbalance(den_tree)
    #print(f'M: {num_nodes}\tN: {num_leaves}\tMax depth: {max_depth}\tColless imbalance: {imb}')
    return max_depth,imb


# these changes did basically nothing to runtime :/

target_depth = 13.0
target_imb = 0.21
n_samp = 1000

birth_rate = 1.5
death_rate = 0.586


def test2():
    dpths = np.empty(n_samp)
    imbs = np.empty(n_samp)
    devs = np.empty(100)
    for j in range(100):
        for i in range(n_samp):
            g_tree = ngesh.gen_tree(num_leaves=48, birth=birth_rate, death=death_rate)
            dpth, imb = analyze_tree(g_tree)
            dpths[i] = dpth
            imbs[i] = imb
        mean_depth = np.mean(dpths)
        mean_imb = np.mean(imbs)
        dev = abs(target_depth - mean_depth) + abs(target_imb - mean_imb)
        devs[j] = dev
        print(dev)
    print('\n',np.mean(devs))

def test1():
    print(f'birth_rate:\t{birth_rate}')
    dpths = np.empty(n_samp)
    imbs = np.empty(n_samp)

    rates = np.arange(0.37,0.38,0.001)
    R = len(rates)
    devs = np.empty(R)
    for j in range(R):
        death_rate = rates[j]
        for i in range(n_samp):
            g_tree = ngesh.gen_tree(num_leaves=48, birth=birth_rate, death=death_rate, seed=i)
            dpth, imb = analyze_tree(g_tree)
            dpths[i] = dpth
            imbs[i] = imb

        mean_depth = np.mean(dpths)
        mean_imb = np.mean(imbs)
        dev = abs(target_depth - mean_depth) + abs(target_imb - mean_imb)
        devs[j] = dev
        print(f'{death_rate:0.3f}\t{dev:0.3f}')






