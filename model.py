from __future__ import print_function

from util import *
import sys

# Parameters
learning_rate = 0.001
#training_iters = 100000
training_iters = 10000
batch_size = 1
display_step = 10
vocab_size = 15000
EARLY_STOP_PATIENCE = 500

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2
n_layers = 2

# tf Graph input
#x = tf.placeholder("float", [batch_size, n_steps, n_input])
#x = tf.placeholder("float", [batch_size, None, n_input])
#y = tf.placeholder("float", [batch_size, n_classes])
x = tf.placeholder(tf.int32, [1, None])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
    #'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases, vocab_size):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    #print('x')
    #print(x.get_shape().as_list())

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * n_layers, state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * n_layers, state_is_tuple=True)

    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, n_hidden], initializer=tf.random_uniform_initializer(-1.0, 1.0))
        #embedding = tf.get_variable("embedding", [vocab_size, n_hidden])
        #embedding = tf.Variable(tf.random_uniform([vocab_size, n_hidden], -1.0, 1.0))
        rnn_inputs = tf.nn.embedding_lookup(embedding, x)

    #outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    #outputs, _ = tf.nn.dynamic_rnn(lstm_fw_cell, x, dtype=tf.float32)
    #outputs, _ = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    with tf.variable_scope("bidirectional_rnn"):
        # Forward direction
        with tf.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = tf.nn.dynamic_rnn(lstm_fw_cell, rnn_inputs, dtype=tf.float32)
            #print('output_fw')
            #print(output_fw.get_shape().as_list())
        with tf.variable_scope("bw") as bw_scope:
            tmp = array_ops.reverse(rnn_inputs, [False, True, False])
            output_bw, output_state_bw = tf.nn.dynamic_rnn(lstm_bw_cell, tmp, dtype=tf.float32)
            output_bw = array_ops.reverse(output_bw, [False, True, False])
    rnn_outputs = tf.concat(2, [output_fw, output_bw])

    #outputs = []
    #fw_state = lstm_fw_cell.zero_state(batch_size, tf.float32)
    #bw_state = lstm_bw_cell.zero_state(batch_size, tf.float32)

    #for idx, input in enumerate(x):
    #    with tf.variable_scope("RNN_FW"):
    #        if idx > 0: tf.get_variable_scope().reuse_variables()
    #        fw_output, fw_state = lstm_fw_cell(input, fw_state)
    #    with tf.variable_scope("RNN_BW"):
    #        if idx > 0: tf.get_variable_scope().reuse_variables()
    #        bw_output, bw_state = lstm_bw_cell(input, bw_state)
    #    print(fw_output.get_shape().as_list())
    #    print(bw_output.get_shape().as_list())
    #    # outputs.append(np.concatenate((fw_output, bw_output), axis=1))
    #    outputs.append(array_ops.concat(1, [fw_output, bw_output]))


    #print("rnn_outputs:")
    #print(rnn_outputs.get_shape().as_list())

    #rnn_outputs = tf.squeeze(rnn_outputs)
    rnn_outputs = tf.reshape(rnn_outputs, [-1, 2 * n_hidden])
    #print("concat:")
    #print(rnn_outputs.get_shape().as_list())

    outputs = tf.matmul(rnn_outputs, weights['out']) + biases['out']

    return embedding, rnn_inputs, rnn_outputs, outputs

embedding, rnn_inputs, rnn_outputs, outputs = BiRNN(x, weights, biases, vocab_size)

pred = tf.nn.softmax(outputs)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = -tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

regularizers = tf.nn.l2_loss(weights['out'])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, y)) + 1e-7 * regularizers

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
# init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

train_data, train_label = generate_data_set("./keyword/train")
test_data, test_label = generate_data_set("./keyword/test")
train_data_set, count, dictionary, reverse_dictionary = build_fixed_size_dataset(train_data, vocab_size)
test_data_set = build_test_set(test_data, dictionary)

train_data_len = len(train_label)
print('train data len: %d' % train_data_len)


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    index = 0

    # early stopping
    best_valid = np.inf
    best_valid_epoch = 0

    # Keep training until reach max iterations
    while step < training_iters:
        batch_x = train_data_set[index]
        #print("batch_x shape:")
        #print(np.shape(batch_x))
        batch_y = train_label[index]
        #print("batch_y shape:")
        #print(np.shape(batch_y))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            emb, rnn_in, rnn_out, prediction, acc, _loss, _cost = sess.run([embedding, rnn_inputs, rnn_outputs, pred, accuracy, loss, cost], feed_dict={x: batch_x, y: batch_y})
            #print('emb dict')
            #print(emb)
            #print('rnn input')
            #print(rnn_in)
            #print('rnn out')
            #print(rnn_out)
            #print('prediction')
            #print(prediction)
            #print('loss')
            #print(_loss)
            #print('cost')
            #print(_cost)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

            # Should use validation loss, but data set is too small
            if _loss < best_valid:
                best_valid = _loss
                best_valid_epoch = step
            elif best_valid_epoch + EARLY_STOP_PATIENCE < step:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch))
                break

        step += 1
        index = (index + 1) % train_data_len
        sys.stdout.flush()
    print("Optimization Finished!")

    test_label_set = sess.run(pred, feed_dict={x:test_data_set[0]})
    print('test data:')
    print(test_data[0])
    #print('test label:')
    #print(test_label_set)
    key_words = generate_key_word(test_data[0], test_label_set)
    print(key_words)

    print("Test Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data_set[0], y: test_label[0]}))
