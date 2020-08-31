import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import io

from tf_tcmc.tcmc.tcmc import TCMCProbability


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





def amino_acid_probability_distribution(tcmc_layer, t, model_id):
    
    
    P = tcmc_layer.probability_distribution(t)[model_id,...].numpy().T
    pi = tcmc_layer.stationary_distribution[model_id,...]


    codons = itertools.product(*["acgt" for i in range(3)])
    codons = [''.join(c) for c in codons]




    aa_order = [v for k in amino_acids for v in amino_acids[k]]
    perm = [codons.index(c) for c in aa_order]


    P_perm = P[np.array(perm)[:,None], perm]
    pi_perm = pi.numpy()[perm]

    num_aa = len(amino_acids)
    A = np.zeros((num_aa,num_aa))

    for i,a in enumerate(amino_acids):
        for j,b in enumerate(amino_acids):

            a_indices = np.array([aa_order.index(c) for c in amino_acids[a]])
            b_indices = np.array([aa_order.index(c) for c in amino_acids[b]])

            P_BA = P_perm[b_indices[:,None], a_indices]
            pi_A = pi_perm[a_indices]

            A[i,j] = (1/np.sum(pi_A)) * np.sum(np.dot(P_BA, pi_A))
    
    return A





def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    
    return image


def plot_amino_acid_probability_distribution(tcmc, model_id=0, t=1):
    
    aa = amino_acids

    A = amino_acid_probability_distribution(tcmc, t, model_id)

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(15, 10), dpi=187)
    ax = sns.heatmap(A, annot=True, fmt=".2f", linewidths=.5, xticklabels=list(aa.keys()), yticklabels=list(aa.keys()), ax=ax)
    
    return f



def plot_generator(tcmc, model_id=0):
    Q = tcmc.generator[0,...]
    
    codons = itertools.product(*["acgt" for i in range(3)])
    codons = [''.join(c) for c in codons]
    
    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(54, 36), dpi=187)
    ax = sns.heatmap(Q, annot=True, fmt=".2f", linewidths=.5, xticklabels=codons, yticklabels=codons, ax=ax)
    
    return f





def log_amino_acid_probability_distribution(epoch, logs, tcmc, file_writer, model_id, t):
    
    figure = visualization.plot_amino_acid_probability_distribution(tcmc, model_id, t)
    image = visualization.plot_to_image(figure)
    
    # Log the confusion matrix as an image summary.
    with file_writer.as_default():
        tf.summary.image(f"Amino Acid Probability Matrix at t={t} for TCMC Model {model_id}", image, step=epoch)




def log_generator(epoch, logs, tcmc, file_writer, model_id):
    
    figure = visualization.plot_generator(tcmc, model_id)
    image = visualization.plot_to_image(figure)
    
    # Log the confusion matrix as an image summary.
    with file_writer.as_default():
        tf.summary.image(f"Generator Q-Matrix for TCMC Model {model_id}", image, step=epoch)




