from os import path
import tensorflow_datasets as tfds
import tensorflow as tf

def load_data(batch_size: int, train_size: float, data_dir: path) -> tuple(tf.data.Dataset):

    split = [f'train[:{train_size*100:.0f}%]', #train
             f'train[{train_size*100:.0f}%]:', #val
             f'test']                          #test

    train, val, test = tfds.load(name='imdb_reviews',
                                 data_dir=data_dir,
                                 split=split,
                                 with_info=False, 
                                 batch_size=batch_size, 
                                 try_gcs=True) 

    return train, val, test
