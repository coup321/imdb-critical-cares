from os import path
import tensorflow_datasets as tfds
import tensorflow as tf

def load_data(batch_size: int, train_size: float, split: list[str], data_dir: path) -> tuple(tf.data.Dataset):

    train, val, test = tfds.load(
        name='imdb_reviews',
        data_dir=data_dir,
        with_info=False, 
        batch_size=batch_size, 
        try_gcs=True) 

    return (train, val, test)
