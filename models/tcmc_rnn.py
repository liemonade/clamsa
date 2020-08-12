import tensorflow as tf
import sys
sys.path.append("..")


from utilities import database_reader
from tf_tcmc.tcmc.tcmc import TCMCProbability
from tf_tcmc.tcmc.tensor_utils import BatchedSequences


def create_model(forest, 
                 alphabet_size,
                 tcmc_models=8,
                 rnn_type='lstm',
                 rnn_units=32,
                 dense_dimension=16,
                 name="aladdin_tcmc_rnn"):
    
    num_leaves = database_reader.num_leaves(forest)
    N = max(num_leaves)
    s = alphabet_size
    
    # define the inputs
    sequences = tf.keras.Input(shape=(N,s), name = "sequences", dtype=tf.float64)
    clade_ids = tf.keras.Input(shape=(), name = "clade_ids", dtype=tf.int32)
    sequence_lengths = tf.keras.Input(shape = (1,), name = "sequence_lengths", dtype = tf.int64) # keras inputs doesn't allow shape [None]


    # define the layers
    tcmc_layer = TCMCProbability((tcmc_models,), forest, name="P_sequence_columns")
    log_layer = tf.keras.layers.Lambda(tf.math.log, name="log_P", dtype=tf.float64)
    bs_layer = BatchedSequences(feature_size = tcmc_models, dtype=tf.float64, name="padded_batched_log_P")    
    
    rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units), name="lstm", dtype=tf.float64)
    if rnn_type == "gru":
        rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_units), name=rnn_type, dtype=tf.float64)
        
    dense_layer = tf.keras.layers.Dense(dense_dimension, kernel_initializer = "TruncatedNormal", activation = "sigmoid", name="dense")
    guesses_layer = tf.keras.layers.Dense(2, kernel_initializer = "TruncatedNormal", activation = "softmax", name = "guesses", dtype=tf.float64)
    
    
    # assemble the computational graph
    P = tcmc_layer(sequences, clade_ids)
    log_P = log_layer(P) 
    batched_log_P = bs_layer([log_P, sequence_lengths])
    rnn_P = rnn_layer(batched_log_P)
    dense = dense_layer(rnn_P)
    guesses = guesses_layer(dense)

    model = tf.keras.Model(inputs = [sequences, clade_ids ,sequence_lengths], outputs = guesses, name = name)
    
    return model


def training_callbacks():
    return []