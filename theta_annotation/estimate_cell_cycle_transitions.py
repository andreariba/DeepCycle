
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import interp1d
import anndata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import argparse
import anndata
import json
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='Estimate cell cycle phase transitions.')
parser.add_argument('--input_adata',type=str, required=True,help='Anndata input file preprocessed with velocyto, scvelo (moments) and DeepCycle.')
parser.add_argument('--gene_phase_dict',type=str, required=True,help='Dictionary containing the list of genes associated with S and G2M phases.')
parser.add_argument('--output_npy_transitions',type=str, required=True,help='Output npy array with thetas in the order: G1/S, M/G1, and possible S/G2.')
parser.add_argument('--output_svg_plot',type=str, required=True,help='SVG plot with all the scores and transitions.')
args = parser.parse_args()


input_adata_file = args.input_adata
gene_phase_dict_filename = args.gene_phase_dict
output_npy_transitions = args.output_npy_transitions
output_svg_plot = args.output_svg_plot

sys.stdout.write("[Loading phase genes]: ")
sys.stdout.flush()
with open(gene_phase_dict_filename, 'r') as jsonfile:
    gene_phase_dict = json.load(jsonfile)
print(gene_phase_dict_filename)


sys.stdout.write("[Loading anndata]: ")
sys.stdout.flush()
adata = anndata.read_h5ad(input_adata_file)
print(input_adata_file)


def add_cell_cycle_scores(adata, gene_phase_dict):
    G1S_total = np.array([])
    number_G1S_genes = len(gene_phase_dict['G1/S'])
    for gene in gene_phase_dict['G1/S']:
        try:
            G1S_total = np.append(G1S_total, scipy.stats.zscore(adata[:,gene].layers['Ms']))
        except KeyError:
            number_G1S_genes -= 1
            continue

    G1S_total = G1S_total.reshape(number_G1S_genes,-1)

    G2M_total = np.array([])
    number_G2M_genes = len(gene_phase_dict['G2/M'])
    for gene in gene_phase_dict['G2/M']:
        try:
            G2M_total = np.append(G2M_total, scipy.stats.zscore(adata[:,gene].layers['Ms']))
        except KeyError:
            number_G2M_genes -= 1
            continue
    G2M_total = G2M_total.reshape(number_G2M_genes,-1)
    
    CyclinE_genes = ['CCNE1','CCNE2','Ccne1','Ccne2']
    CyclinE_total = np.array([])
    number_CyclinE_genes = len(CyclinE_genes)
    for gene in CyclinE_genes:
        try:
            CyclinE_total = np.append(CyclinE_total, scipy.stats.zscore(adata[:,gene].layers['Ms']))
        except KeyError:
            number_CyclinE_genes -= 1
            continue
    CyclinE_total = CyclinE_total.reshape(number_CyclinE_genes,-1)

    G1S_score = G1S_total.mean(axis=0)
    G2M_score = G2M_total.mean(axis=0)
    CyclinE_score = CyclinE_total.mean(axis=0)
    
    adata.obs['S_score'] = G1S_score
    adata.obs['G2M_score'] = G2M_score
    adata.obs['CyclinE_score'] = CyclinE_score



def estimate_phase_transitions(adata):
    
    theta = adata.obs['cell_cycle_theta']
    
    #create array to calculate the moving average
    bins = np.linspace(min(theta),max(theta),50)
    selecting_array = []
    x = []
    for i in range(len(bins)-1):
        b0 = bins[i]
        bf = bins[i+1]
        x.append((bins[i]+bins[i+1])/2)
        vi = []
        for t in theta:
            if t<bf and t>=b0:
                vi.append(1.0)
            else:
                vi.append(0.0)
        vi = np.array(vi)
        if sum(vi)>0:
            vi /= sum(vi)
        selecting_array.append(vi)
    selecting_array = np.array(selecting_array)
    
    S_score = adata.obs['S_score']
    G2M_score = adata.obs['G2M_score']
    n_counts = adata.obs['n_counts']/1000
    CyclinE_score = adata.obs['CyclinE_score']
    
    S_score_avg = selecting_array.dot(S_score)
    G2M_score_avg = selecting_array.dot(G2M_score)
    n_counts_avg = selecting_array.dot(n_counts)
    CyclinE_score_avg = selecting_array.dot(CyclinE_score)
    
    theta = x
    
    fig = plt.figure(figsize=plt.figaspect(.5))

    theta_G1S = round(theta[ 1+np.where(CyclinE_score_avg==CyclinE_score_avg.max())[0][0] ],2)
    theta_M = round(theta[ 1+np.where(n_counts_avg==n_counts_avg.max())[0][0] ],2)
    theta_SG2 = []
    f = scipy.interpolate.splrep(x=theta,y=G2M_score_avg-S_score_avg,k=3)
    roots = scipy.interpolate.sproot(f)
    for root in roots:
        der_value = scipy.interpolate.splev(root,f,der=1)
        if der_value>=0:
            theta_SG2.append( round( theta[ np.where( np.abs(theta-root)==np.abs(theta-root).min())[0][0] ],2) )

    saved_SG2 = []
    if theta_G1S>theta_M:
        for temp in theta_SG2:
            if temp<theta_M or temp>theta_G1S:
                saved_SG2.append(temp)
    else:
        for temp in theta_SG2:
            if temp<theta_M and temp>theta_G1S:
                saved_SG2.append(temp)
    theta_SG2 = saved_SG2
    thetas_to_plot = [theta_G1S, theta_M]+ theta_SG2

    ax = fig.add_subplot(1,2,1,projection='3d')
    for i in range(len(theta)):
        ax.plot(S_score_avg[i:i+2], G2M_score_avg[i:i+2], n_counts_avg[i:i+2], color=cm.plasma(theta[i]))
    for i in range(len(theta)):
        if round(theta[i],2) in thetas_to_plot:
            ax.scatter(S_score_avg[i],G2M_score_avg[i],n_counts_avg[i],color='red')
            ax.text(S_score_avg[i],G2M_score_avg[i],n_counts_avg[i],str(round(theta[i],2)))
    ax.set(xlabel='S score',ylabel='G2M score',zlabel='UMI counts (x1k)')

    ax = fig.add_subplot(2,2,2)
    for i in range(len(theta)):
        ax.plot(S_score_avg[i:i+2], CyclinE_score_avg[i:i+2], color=cm.plasma(theta[i]))
    for i in range(len(theta)):
        if round(theta[i],2) in thetas_to_plot:
            ax.scatter(S_score_avg[i],CyclinE_score_avg[i],color='red')
            ax.text(S_score_avg[i],CyclinE_score_avg[i],str(round(theta[i],2)))
    ax.plot([-1.0,1.5],[-1,1.5],linestyle='dashed',color='gray')
    ax.set_aspect(1.0)
    ax.set(xlabel='S score',ylabel='CyclinE score')

    ax = fig.add_subplot(2,2,4)
    for i in range(len(theta)):
        ax.plot(S_score_avg[i:i+2], G2M_score_avg[i:i+2], color=cm.plasma(theta[i]))
    for i in range(len(theta)):
        if round(theta[i],2) in thetas_to_plot:
            ax.scatter(S_score_avg[i],G2M_score_avg[i],color='red')
            ax.text(S_score_avg[i],G2M_score_avg[i],str(round(theta[i],2)))
    ax.plot([-1.0,1.5],[-1.0,1.5],linestyle='dashed',color='gray')
    ax.set_aspect(1.0)
    ax.set(xlabel='S score',ylabel='G2M score')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.025, 0.5])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
    fig.colorbar(mapper,cax=cbar_ax)

    return thetas_to_plot

# Estimation 

add_cell_cycle_scores(adata, gene_phase_dict)

theta_transitions = estimate_phase_transitions(adata)
plt.savefig(output_svg_plot)

np.save("theta_transitions.npy",theta_transitions)
print("[G1/S]:",theta_transitions[0])
print("[S/G2]:",theta_transitions[1])
print("[M/G1]:",theta_transitions[2:])

print("[Theta saved to]:", output_npy_transitions)


