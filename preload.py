import numpy as np
from ete3 import Tree
import pandas as pd
import ngesh

batch_size = 64
sample_size = 128
ns = [round(n) for n in np.logspace(start=3, stop=10, num=sample_size, base=2)]

org_path = "./preload/original_data/"
gen_path = "./preload/generated_data/"


def generate_sample(n=48,d=200,seed=0):
    print("generating sample")
    # birth/death rates chosen to generate tree similar to source
    tree = ngesh.gen_tree(num_leaves=n,birth=1.5,death=0.586,seed=seed)
    x = np.random.normal(size=(n,d))
    preorder = [n for n in tree.traverse('preorder')]
    leaves = [n for n in preorder if n.is_leaf()]
    return x, preorder, leaves, tree

def generate_data():
    print("Generating data")
    for n in ns:
        print(f'n={n}')
        for s in range(batch_size):
            print(f' seed={s}')
            X, Preorder, Leaves, Ptree = generate_sample(n, seed=s)
            pth = f'./preload/generated_data/{n:0>4}_{s:0>2}_'
            np.savez(pth + "arrays", X, Preorder, Leaves)
            Ptree.write(format=1, outfile=pth + "tree.nw")

def process_original_data():
    data = np.loadtxt("/var/home/luka/proj/Papilonidae_dataset_v2/Papilionidae_aligned_new.txt", delimiter="\t").reshape((2240, 200))
    cats = pd.read_csv("/var/home/luka/proj/Papilonidae_dataset_v2/Papilonidae_metadata_new.txt", header=None)[0]

    tree = Tree("/var/home/luka/proj/Papilonidae_dataset_v2/papilionidae_tree.txt", format=3)
    leaf_names = tree.get_leaf_names()  # returns in-order leaves (same as pre-order when just looking at leaves)
    preorder = [n for n in tree.traverse('preorder')]
    leaves = [n for n in preorder if n.is_leaf()]

    df = pd.DataFrame(data.reshape(2240, -1))
    df['category'] = pd.Categorical(cats, categories=cats.unique())
    means = df.groupby('category', observed=True).mean()
    df['order'] = pd.Categorical(df['category'], categories=leaf_names, ordered=True)
    means = means.reset_index()
    means['order'] = pd.Categorical(means['category'], categories=leaf_names, ordered=True)
    means = means.sort_values('order')
    x = means.drop(columns=['category', 'order']).values

    np.savez("./preload/original_data/arrays",x,leaves,preorder)
    tree.write(format=1,outfile="./preload/original_data/tree.nw")

def load_data(path):
    arrs = np.load(path + "arrays.npz", allow_pickle=True)
    [x, preorder, leaves] = [arrs[f'arr_{i}'] for i in range(3)]
    tree = Tree(path + "tree.nw", format=1)
    return x, preorder, leaves, tree

def load_original_data():
    return load_data(org_path)

def load_generated_data(size=0,seed=0):
    n = ns[size]
    path = gen_path+f'{n:0>4}_{seed:0>2}_'
    return load_data(path)
