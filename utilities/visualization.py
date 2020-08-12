import sys
sys.path.append("..")
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from tf_tcmc.tcmc.tcmc import TCMCProbability





def amino_acid_probability_distribution_heatmap(tcmc, t, model_id, ax):
    # TODO: Rewrite code for the general case
    #sns.set()

    P = tcmc.probability_distribution(1)[model_id,...].numpy().T
    pi = tcmc.stationary_distribution[model_id,...]


    codons = itertools.product(*["acgt" for i in range(3)])
    codons = [''.join(c) for c in codons]


    amino_acids = {
        'M' : ['atg',],
        'W' : ['tgg',],
        'Y' : ['tat', 'tac',],
        'F' : ['ttt', 'ttc',],
        'C' : ['tgt', 'tgc',],
        'N' : ['aat', 'aac',],
        'D' : ['gat', 'gac',],
        'Q' : ['caa', 'cag',],
        'E' : ['gaa', 'gag',],
        'H' : ['cat', 'cac',],
        'K' : ['aaa', 'aag',],
        'I' : ['att', 'atc', 'ata'],
        'G' : ['ggt', 'ggc', 'gga', 'ggg',],
        'A' : ['gct', 'gcc', 'gca', 'gcg',],
        'V' : ['gtt', 'gtc', 'gta', 'gtg',],
        'T' : ['act', 'acc', 'aca', 'acg',],
        'P' : ['cct', 'ccc', 'cca', 'ccg',],
        'L' : ['ctt', 'ctc', 'cta', 'ctg', 'tta', 'ttg',],
        'S' : ['tct', 'tcc', 'tca', 'tcg', 'agt', 'agc',],
        'R' : ['cgt', 'cgc', 'cga', 'cgg', 'aga', 'agg',],
        '<' : ['taa', 'tag', 'tga',],
    }


    aa_order = [v for k in aa for v in amino_acids[k]]
    perm = [codons.index(c) for c in aa_order]


    P_perm = P[np.array(perm)[:,None], perm]
    pi_perm = pi.numpy()[perm]

    num_aa = len(amino_acids)
    A = np.zeros((num_aa,num_aa))

    for i,a in enumerate(amino_acids):
        for j,b in enumerate(aa):

            a_indices = np.array([aa_order.index(c) for c in amino_acids[a]])
            b_indices = np.array([aa_order.index(c) for c in amino_acids[b]])

            P_BA = P_perm[b_indices[:,None], a_indices]
            pi_A = pi_perm[a_indices]

            A[i,j] = (1/np.sum(pi_A)) * np.sum(np.dot(P_BA, pi_A))

    print('Sanity check for A matrix (col sums):')
    print(np.sum(A,axis=1))