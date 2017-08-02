
# coding: utf-8

# In[1]:

import os
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from scipy import ndimage
import tensorflow as tf
import tarfile

TRAIN_SIZE = 650
TEST_SIZE = 64


NUM_IMAGES = 714
IMAGE_SIZE = 32
NUM_CHANNELS = 4
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS

TYPE_ARRAY=np.array(['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy',  
                    'Fighting', 'Fire', 'Ghost', 'Grass', 'Ground',  
                    'Ice','Normal','Poison','Psychic','Rock','Steel','Water'])\

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def extract_images(tfile):
    print("Extracting " + tfile)
    tar = tarfile.open(tfile, "r:gz")
    tar.extractall(DATA_DIR)
    print ("Extracted to " + DATA_DIR)
    tar.close()

def get_directories():
    datadir = os.environ["KRYLOV_DATA_DIR"]
    principal = os.environ["KRYLOV_WF_PRINCIPAL"]

    DATA_DIR = os.path.join(datadir , principal)
    TAR_FILE = os.path.join(DATA_DIR , 'data.tar.gz')
    IMAGE_DIRECTORY = os.path.join (DATA_DIR , 'data')
    return DATA_DIR, TAR_FILE, IMAGE_DIRECTORY

        
def load_images(image_dir):
    labels = []
    names = []
    image_index = 0
    
    
    # 714 because the Flying Pokemon were removed
    images = np.ndarray(shape=(714, IMAGE_ARR_SIZE))
                        
    
    # Loop through all the types directories
    for type in os.listdir(image_dir):
        type_images = os.listdir(os.path.join(image_dir , type))
        
        # Loop through all the images of a type directory
        for image in type_images:
            image_file = os.path.join(image_dir, type, image)
            names.append(image)

            # reading the images as they are; no normalization, no color editing
            image_data = mpimg.imread(image_file) 
            #image_data = image_data.mean(axis=2).astype(np.float32)
            images[image_index, :] = image_data.flatten()
            image_index += 1
            labels.append(type)
        
    return (images, np.asarray(labels), np.asarray(names))



def get_pokemon_of_type(X, labels, type):
    poke_type = []
    for row in range(len(X)-1):
        if (labels[row]==type):
            poke_type.append(X[row])
    return np.asarray(poke_type)

def get_pokemon_type_index(type_name):
    for i in range(len(TYPE_ARRAY)):
        if TYPE_ARRAY[i] == type_name:
            return i;
    return 0

def get_labels_index(labels):
    labels_index = []
    for i in range(len(labels)):
        labels_index.append(get_pokemon_type_index(labels[i]))
    return labels_index  


# In[2]:

height = IMAGE_SIZE
width = IMAGE_SIZE
channels = NUM_CHANNELS
n_inputs = height * width * channels

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 17

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 8 * 8])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# In[3]:

DATA_DIR, TAR_FILE, IMAGE_DIRECTORY = get_directories()
print (DATA_DIR, TAR_FILE, IMAGE_DIRECTORY)

#extract_images(TAR_FILE)
dataset, labels, names = load_images(IMAGE_DIRECTORY)
shuffle_index = np.random.permutation(NUM_IMAGES)
dataset, labels = dataset[shuffle_index], labels[shuffle_index]
labels_index = get_labels_index(labels)
X_train, labels_train, labels_index_train = dataset[:650], labels[:650], labels_index[:650]
X_test, labels_test = dataset[650:], labels_index[650:]

def get_next_batch(cursor, batchsize=50):
    if (cursor >= TRAIN_SIZE):
        cursor = 0
    X_batch, y_batch = X_train[cursor:cursor+batchsize], labels_index[cursor:cursor+batchsize]
    cursor = cursor + batchsize
    return cursor, X_batch, np.array(y_batch)



# In[ ]:

n_epochs = 10
batch_size = 10

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        cursor = 0
        for iteration in range(TRAIN_SIZE // batch_size):
            cursor, X_batch, y_batch = get_next_batch(batch_size, cursor)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
       
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: labels_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./my_mnist_model")
        
    for i in range(len(TYPE_ARRAY)):
        poke = get_pokemon_of_type(dataset[650:], labels[650:], TYPE_ARRAY[i])
        poke_labels = np.full(len(poke), i)
        poke_acc = accuracy.eval(feed_dict={X: poke, y: poke_labels})
        print (TYPE_ARRAY[i], len(poke), "Accuracy:", poke_acc)




