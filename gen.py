import numpy as np
import ngesh

sample_size = 12
ns = [round(n) for n in np.logspace(start=3,stop=10,num=50,base=2)]

def generate_data(n,d=200,seed=0):
    tree = ngesh.gen_tree(num_leaves=n,seed=seed)
    x = np.random.normal(size=(n,d))
    preorder = [n for n in tree.traverse('preorder')]
    leaves = [n for n in preorder if n.is_leaf()]
    return x, preorder, leaves, tree

print("Generating data")
for n in ns:
    print(f'n={n}')
    for s in range(sample_size):
        print(f' seed={s}')
        X, Preorder, Leaves, Ptree = generate_data(n,seed=s)
        pth = f'./preload/generated_data/{n:0>4}_{s:0>2}_'
        np.savez(pth+"arrays", X, Leaves, Preorder)
        Ptree.write(format=1, outfile=pth+"tree.nw")
