import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
import pickle
import time

int_text, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))

save_dir = './save'

## Hyperparameters ##

# Number of Epochs
num_epochs = 1000
# Batch Size
batch_size = 50
# RNN Size
rnn_size = 512
# Embedding Dimension Size
embed_dim = 200
# Sequence Length
seq_length = 20
# Learning Rate
learning_rate = 0.001
# RNN Layer Number
rnn_layers_num = 2
# Dropout Keep Probability
keep_probability = 0.75
# Show stats for every n number of batches
show_every_n_batches = 1000

## Input ##

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learing_rate')
    return inputs, targets, learning_rate


## Word Embedding ##

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


## Build RNN Cell and Initialize ##

def get_init_cell(batch_size, rnn_size, num_layers, keep_prob):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # Define Single LSTM layer
    def single_cell():
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        # Add dropout to the cell
        return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    # Stack up multiple LSTM layers, for deep learning
    multi_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)], state_is_tuple=True)
    initial_state = multi_cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')

    return multi_cell, initial_state


## Build RNN ##

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(state, name='final_state')

    return outputs, final_state


## Build the Neural Network ##

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # Embedding Layer
    embeded = get_embed(input_data, vocab_size, embed_dim)

    # RNNs Layer
    rnn_output, final_state = build_rnn(cell, embeded)

    # Full Connected Layer
    # DIY
    #     rnn_output = tf.concat(rnn_output, axis=1)
    #     rnn_output = tf.reshape(rnn_output, [-1, rnn_size])

    #     with tf.variable_scope('fc_layer'):
    #         fc_weights = tf.Variable(tf.truncated_normal((rnn_size, vocab_size), stddev=0.01))
    #         fc_bias = tf.Variable(tf.zeros(vocab_size))

    #     logits = tf.matmul(rnn_output, fc_weights) + fc_bias
    #     logits = tf.reshape(logits, input_data.get_shape().as_list() + [vocab_size])

    # Using tf.contrib.layers.fully_connected()
    logits = tf.contrib.layers.fully_connected(rnn_output,
                                               vocab_size,
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               activation_fn=None)

    return logits, final_state


## Get Batches ##

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # Calculate number of words of each batch
    batch_words = batch_size * seq_length

    # Calculate number of batches
    batch_num = len(int_text) // batch_words

    # Modify dataset for full batches
    int_text = np.array(int_text[:batch_num * batch_words])
    int_text = int_text.reshape((batch_size, -1))

    batches = np.zeros((batch_num, 2, batch_size, seq_length), dtype=np.int)

    for i in range(0, int_text.shape[1], seq_length):
        n = int(i / seq_length)

        batches[n, 0, :, :] = int_text[:, i:i + seq_length]

        if n != batch_num - 1:
            batches[n, 1, :, :] = int_text[:, i + 1:i + seq_length + 1]
        else:
            batches[n, 1, :, :-1] = int_text[:, i + 1:i + seq_length]
            batches[n, 1, :-1, -1] = int_text[1:, 0]
            batches[n, 1, -1, -1] = int_text[0, 0]

            #     print(int_text)
            #     print(batches)

    return batches


## Build the Graph ##

train_graph = tf.Graph()

with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size, rnn_layers_num, keep_probability)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


## Training Model##

batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})
        start = time.time()

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            batch_num_total = epoch_i * len(batches) + batch_i
            if (batch_num_total) % show_every_n_batches == 0:
                end = time.time()
                print('Epoch: {:>3} | Batch: {:>2}/{} | Batch_Total: {:>5} | {:.6f} Sec/Batch || train_loss = {:.4f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    batch_num_total,
                    (end - start) / show_every_n_batches,
                    train_loss))
                start = time.time()

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


pickle.dump((seq_length, save_dir), open('params.p', 'wb'))
