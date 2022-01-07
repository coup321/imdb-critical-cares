from os import path
import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path

def load_data(batch_size: int, train_size: float, data_dir: path) -> tuple:

    split = [f'train[:{train_size*100:.0f}%]', f'train[{train_size*100:.0f}%:]', f'test']                          

    train, val, test = tfds.load(name='imdb_reviews',
                                 data_dir=data_dir,
                                 split=split,
                                 with_info=False, 
                                 batch_size=batch_size, 
                                 try_gcs=True,
                                 as_supervised=True)     
    
    return train, val, test
