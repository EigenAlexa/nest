from collections import namedtuple
import csv
import re
import string

import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf

# reads in the data
reader = csv.reader(open("wikipedia.csv"))
count = 0
data = ''
for row in reader:
    count = count + 1
    if count > 301:
        break
    else:
        data += row[1].lower()

sentenceEnders = re.compile('[.?!]')
data_list = sentenceEnders.split(data)

LabelDoc = namedtuple('LabelDoc', 'words tags')
exclude = set(string.punctuation)
all_docs = []
count = 0
for sen in data_list:
    word_list = sen.split()
    if len(word_list) < 3:
        continue
    tag = ['SEN_' + str(count)]
    count += 1
    sen = ''.join(ch for ch in sen if ch not in exclude)
    all_docs.append(LabelDoc(sen.split(), tag))


# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(input_data, min_cut_freq):
    words = []
    for i in input_data:
        for j in i.words:
            words.append(j)
    count_org = [['UNK', -1]]
    count_org.extend(collections.Counter(words).most_common())
    count = [['UNK', -1]]
    for word, c in count_org:
        word_tuple = [word, c]
        if word == 'UNK':
            count[0][1] = c
            continue
        if c > min_cut_freq:
            count.append(word_tuple)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for tup in input_data:
        word_data = []
        for word in tup.words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            word_data.append(index)
        data.append(LabelDoc(word_data, tup.tags))
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


min_cut_freq = 3  # cut off frequence smaller then 3 words
data, count, dictionary, reverse_dictionary = build_dataset(all_docs, min_cut_freq)
vocabulary_size = len(reverse_dictionary)
paragraph_size = len(all_docs)
print('paragraph_size: ', paragraph_size)

word_index = 0
sentence_index = 0


def generate_DM_batch(batch_size, num_skips, skip_window):
    global word_index
    global sentence_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    para_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # Paragraph Labels
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[sentence_index].words[word_index])
        sen_len = len(data[sentence_index].words)
        if sen_len - 1 == word_index:  # reaching the end of a sentence
            word_index = 0
            sentence_index = (sentence_index + 1) % len(data)
        else:  # increase the word_index by 1
            word_index += 1
    for i in range(batch_size):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        batch_temp = np.ndarray(shape=(num_skips), dtype=np.int32)
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch_temp[j] = buffer[target]
        batch[i] = batch_temp
        labels[i, 0] = buffer[skip_window]
        para_labels[i, 0] = sentence_index
        buffer.append(data[sentence_index].words[word_index])
        sen_len = len(data[sentence_index].words)
        if sen_len - 1 == word_index:  # reaching the end of a sentence
            word_index = 0
            sentence_index = (sentence_index + 1) % len(data)
        else:  # increase the word_index by 1
            word_index += 1
    return batch, labels, para_labels


# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.
s

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size, skip_window * 2])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # paragraph vector place holder
    train_para_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]: [200, 4] + [vocab_size - 1, embedding_size]
        embed_word = tf.nn.embedding_lookup(embeddings, train_inputs)

        para_embeddings = tf.Variable(
            tf.random_uniform([paragraph_size, embedding_size], -1.0, 1.0))
        embed_para = tf.nn.embedding_lookup(para_embeddings, train_para_labels)

        embed = tf.concat(1, [embed_word, embed_para])

        reduced_embed = tf.div(tf.reduce_sum(embed, 1), skip_window * 2 + 1)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, reduced_embed, train_labels,
                       num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1.0
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.009, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    para_norm = tf.sqrt(tf.reduce_sum(tf.square(para_embeddings), 1, keep_dims=True))
    normalized_para_embeddings = para_embeddings / para_norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels, batch_para_labels = generate_DM_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, train_para_labels: batch_para_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 100000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    final_para_embeddings = normalized_para_embeddings.eval()