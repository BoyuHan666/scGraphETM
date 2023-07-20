import scipy.sparse as sp
import anndata
import pickle

pickle_file_path = '/home/vickarry/projects/ctb-liyue/vickarry/data/tf_gene.pickle'

with open(pickle_file_path, "rb") as file:
    tf_gene = pickle.load(file)


top1 = '/home/vickarry/projects/ctb-liyue/vickarry/data/top1peak_gene_relation.npz'
peak_gene_top1 = sp.load_npz(top1)
correlation1 = peak_gene_top1 + tf_gene
file_path = 'top1_peak_tf_gene.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(correlation1, file)

top3 = '/home/vickarry/projects/ctb-liyue/vickarry/data/top3peak_gene_relation.npz'
peak_gene_top3 = sp.load_npz(top3)
correlation2 = peak_gene_top3 + tf_gene
file_path = 'top3_peak_tf_gene.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(correlation2, file)

top5 = '/home/vickarry/projects/ctb-liyue/vickarry/data/top5peak_gene_relation.npz'
peak_gene_top5 = sp.load_npz(top5)
correlation3 = peak_gene_top5 + tf_gene
file_path = 'top5_peak_tf_gene.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(correlation3, file)

top5_2000 = '/home/vickarry/projects/ctb-liyue/vickarry/data/2000bp_top5peak_gene_relation.npz'
peak_gene_2000 = sp.load_npz(top5_2000)
correlation4 = peak_gene_2000 + tf_gene
file_path = '2000bp_top5_peak_tf_gene.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(correlation4, file)

