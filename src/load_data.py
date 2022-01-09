from os import path
import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path

def load_data(batch_size, train_size):
    """
    TPU instances do not have access to colab VM local file system. They only have
    access to GCS and the memory cache. Because of this, the dataset needs to be
    loaded directly into memory. This is accomplished by indicating batch_size=-1.
    Otherwise, the function will error.

    Input: 
      batch_size, int           (default: BATCHS_SIZE)
      train_size, float 0.0-1.0 (default: TRAIN_SIZE)
    
    Output: (train, val, test), tf.data.Dataset
    """
    #Allows function to write to local file system and then load to memory
    with tf.device('/job:localhost'):
      ds, info = tfds.load(name='imdb_reviews', with_info=True, batch_size=-1, try_gcs=True)

    #number of rows in un-split dataset (train + val) = 25000 for imdb_reviews
    ds_size = ds['train']['label'].numpy().shape[0]
    #Index number to stop at for training set
    #Index number to start at for validation set
    train_size = int(train_size*ds_size)

    train = tf.data.Dataset.from_tensor_slices(
        (ds['train']['text'].numpy()[:train_size], 
        ds['train']['label'].numpy()[:train_size])).batch(batch_size)
    
    val = tf.data.Dataset.from_tensor_slices(
        (ds['train']['text'].numpy()[train_size:], 
        ds['train']['label'].numpy()[train_size:])).batch(batch_size)

    test = tf.data.Dataset.from_tensor_slices(
        (ds['train']['text'], ds['train']['label'])).batch(batch_size)

    return (train, val, test) 
