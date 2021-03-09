import anndata
import pandas as pd
import numpy as np
import scipy
from scipy import optimize
import multiprocessing as mp
from functools import partial

import argparse


parser = argparse.ArgumentParser(description='Fit ISMARA model.')
parser.add_argument('--input_adata',type=str, required=True,help='Anndata input file preprocessed with velocyto and scvelo (moments).')
parser.add_argument('--TFBS',type=str, required=True,help='Matrix of transcription factor binding sites and promoter associations.')
parser.add_argument('--output_adata',type=str, required=True,help='Anndata output file.')
args = parser.parse_args()

input_anndata = args.input_adata
input_TFBS = args.TFBS
output_anndata = args.output_adata

print("[Input anndata]:", input_anndata)
print("[TFBS matrix]:", input_TFBS)

adata = anndata.read_h5ad(input_anndata)
data_frame_tf_promoter = pd.read_csv(input_TFBS,sep=";")

promoter_list = list(data_frame_tf_promoter.index)
motif_list = list(data_frame_tf_promoter.columns)

adata_genes = list(adata.var.index)

genes = []
for g in adata_genes:
    if g in promoter_list:
        genes.append(g)

adata_gene_indexes = []
for g in genes:
    adata_gene_indexes.append( adata_genes.index(g) )


print("[Number of motifs]:", len(motif_list))
print("[Number of genes]:", len(genes))

data_frame_tf_promoter = data_frame_tf_promoter.loc[genes,:]

N = data_frame_tf_promoter.to_numpy()
Ntilde = N - N.mean(axis=0)
E = adata.layers['Mu'][:, adata_gene_indexes]
Eprime = E - E.mean(axis=0)
Etilde = (Eprime.T - Eprime.mean(axis=1)).T

def l_optimization_func(l, E, N, indices):
    # split by gene E and N
    training_idx, test_idx = indices[:int(len(indices)*0.8)], indices[int(len(indices)*0.8):]
    E_training, E_test = E[:,training_idx], E[:,test_idx]
    N_training, N_test = N[training_idx,:], N[test_idx,:]
    W = N_training.T.dot(N_training) + l*np.identity(N_training.shape[1])
    Astar = np.linalg.inv(W).dot(N_training.T).dot(E_training.T)
    difference = (E_test - N_test.dot(Astar).T).flatten()
    return difference.dot(difference.T)

def optimize_l( E, N, n):
    print("[Optimization",n,"]: started")
    indices = np.random.RandomState(seed=n).permutation(E.shape[1])
    res = optimize.minimize(l_optimization_func, x0=1e-10, args=(E, N, indices))
    if not res.success:
        print("[Optimization",n,"]: failed")
        return None
    else:
        print("[Optimal lambda",n,"]:", res.x)
        return float(res.x)

def fit_A(E, N):
    n_cpu = mp.cpu_count()
    print("[Running on]:",n_cpu,"cpu(s)")
    with mp.Pool( n_cpu ) as pool:
        func = partial(optimize_l, E, N)
        res = pool.map(func, range(n_cpu))
    res = [i for i in res if i]
    res = np.array(res)
    print("[lambdas]:",res)
    print("[Mean lambda]:",np.mean(res))
    W = N.T.dot(N) + np.mean(res)*np.identity(N.shape[1])
    Astar = np.linalg.inv(W).dot(N.T).dot(E.T)
    difference = np.abs(E - N.dot(Astar).T)
    #print(difference.shape)
    Chi_squared = (np.multiply(difference, difference)).mean(axis=1)
    return Astar, W, Chi_squared


A, W, C = fit_A(Etilde,Ntilde)

adata.uns['ISMARA_activities'] = pd.DataFrame(A,index=motif_list,columns=adata.obs.index).transpose()

adata.write_h5ad(output_anndata)
print("[Output anndata]:", output_anndata)

