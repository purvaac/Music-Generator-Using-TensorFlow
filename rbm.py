import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tqdm import tqdm
import midi_manipulation
tf.compat.v1.disable_eager_execution()


def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e           
    return songs

songs = get_songs('Pop_Music_Midi')
print("{} songs processed".format(len(songs)))


#hyperparameters
lowest_note = midi_manipulation.lowerBound
highest_note = midi_manipulation.upperBound
note_range = highest_note - lowest_note
num_timesteps = 15
n_visible = 2 * note_range * num_timesteps
n_hidden = 50

num_epochs = 200 #number of times the entire dataset will be fed for training.
batch_size = 100
lr = 0.005 #learning rate

x = tf.compat.v1.placeholder(tf.float32, [None, n_visible], name="x") # placeholder for input data. 
W = tf.Variable(tf.random.normal([n_visible, n_hidden], mean=0.0, stddev=0.01), name="W") #weight
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh")) #hidden bias
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv")) #visible bias

@tf.function
def sample(probs):
    return tf.math.floor(probs + tf.random.uniform(tf.shape(probs), 0, 1))

@tf.function
def gibbs_sample(k):
    def gibbs_step(count, k, xk):
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
        return count + 1, k, xk

    ct = tf.constant(0)
    _, _, x_sample = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                    gibbs_step, [ct, tf.constant(k), x])
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

x_sample = gibbs_sample(1)
h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder = tf.multiply(lr / size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0] / num_timesteps) * num_timesteps)]
            song = np.reshape(song, [song.shape[0] // num_timesteps, song.shape[1] * num_timesteps])
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                sess.run(updt, feed_dict={x: tr_x})

    sample = sess.run(gibbs_sample(1), feed_dict={x: np.zeros((50, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue
        S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
        midi_manipulation.noteStateMatrixToMidi(S, "generated_chord_{}".format(i))
