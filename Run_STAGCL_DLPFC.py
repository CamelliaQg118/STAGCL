import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import metrics
import scipy as sp
import numpy as np
import torch
import copy
import os
import STAGCL

import utils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


ARI_list = []
random_seed = 42
STAGCL.fix_seed(random_seed)
os.environ['R_HOME'] = 'D:/Software/Code/R/R-4.3.3/R-4.3.3'
os.environ['R_USER'] = 'C:/Users/29461/.conda/envs/STAGCL/Lib/site-packages/rpy2'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

dataset = 'DLPFC'
slice = '151507'
platform = '10X'
file_fold = os.path.join('../Data', platform, dataset, slice)
adata, adata_X = utils.load_data(dataset, file_fold)
df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
adata = utils.label_process_DLPFC(adata, df_meta)

savepath = '../Result/DLPFC/' + str(slice) + '/'

if not os.path.exists(savepath):
    os.mkdir(savepath)
n_clusters = 5 if slice in ['151669', '151670', '151671', '151672'] else 7

adata, adj, edge_index, adj_np, adj_remove_dig = utils.graph_build(adata, adata_X, dataset)

stagcl_net = STAGCL.stagcl(adata.obsm['X_pca'], adata, adj, edge_index, adj_remove_dig, n_clusters, dataset, device=device)

tool = None
if tool == 'mclust':
    emb = stagcl_net.train()
    adata.obsm['STAGCL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STAGCL.mclust_R(adata, n_clusters, use_rep='STAGCL', key_added='STAGCL', random_seed=random_seed)
elif tool == 'leiden':
    emb = stagcl_net.train()
    adata.obsm['STAGCL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STAGCL.leiden(adata, n_clusters, use_rep='STAGCL', key_added='STAGCL', random_seed=random_seed)
elif tool == 'louvain':
    emb = stagcl_net.train()
    adata.obsm['STAGCL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STAGCL.louvain(adata, n_clusters, use_rep='STAGCL', key_added='STAGCL', random_seed=random_seed)
else:
    emb, idx = stagcl_net.train()
    print("emb", emb)
    adata.obsm['STAGCL'] = emb
    adata.obs['STAGCL'] = idx
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

print("adata", adata)
new_type = utils.refine_label(adata, radius=15, key='STAGCL')
adata.obs['STAGCL'] = new_type
ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['STAGCL'])
NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['STAGCL'])
adata.uns["ARI"] = ARI
adata.uns["NMI"] = NMI
print('===== Project: {}_{} ARI score: {:.4f}'.format(str(dataset), str(slice), ARI))
print('===== Project: {}_{} NMI score: {:.4f}'.format(str(dataset), str(slice), NMI))
print(str(slice))
print(n_clusters)
ARI_list.append(ARI)

#
fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
sc.pl.spatial(adata, color='ground_truth', ax=axes[0], show=False)
sc.pl.spatial(adata, color=['STAGCL'], ax=axes[1], show=False)
axes[0].set_title("Manual annotation (" + dataset + "#" + slice + ")")
axes[1].set_title('STAGCL_Clustering: (ARI=%.4f)' % ARI)


plt.subplots_adjust(wspace=0.5)  
plt.subplots_adjust(hspace=0.5)  
plt.savefig(savepath + 'STAGCL.jpg', dpi=300)  

