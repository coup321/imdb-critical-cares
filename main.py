from src.train_model import adam_w_optimizer
from src.train_model import load_tpu
from src.load_data import load_data
from models.BERT_LR_Classifier import BERT_LR_Preprocesser, BERT_LR_Classifier
from pathlib import Path
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 
import os

EPOCHS = 2
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
TRAIN_SIZE = 0.6
TEST = False
DATA_DIR = Path("./data/raw")
LOG_DIR = "./logs"
PREPROCESSER = 'BERT_LR_Preprocesser'
MODEL = 'BERT_LR_Classifier'
USE_TPU = False
AUTOTUNE = tf.data.AUTOTUNE

tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/google/experts/bert/pubmed/2'


preprocesser_dict = {
    'BERT_LR_Preprocesser' : BERT_LR_Preprocesser([], 128, tfhub_handle_preprocess)
}

models = {
    'BERT_LR_Classifier' : BERT_LR_Classifier(tfhub_handle_encoder)
}
def main():
    #load data
    train, val, test = load_data(batch_size=BATCH_SIZE,train_size=TRAIN_SIZE, data_dir=DATA_DIR)
    if TEST:
        train = test

    #preprocess data
    preprocesser = preprocesser_dict[PREPROCESSER]
    processed_train = train.map(lambda x, y: (preprocesser(x), y)).cache().prefetch(AUTOTUNE)
    processed_val = val.map(lambda x, y: (preprocesser(x), y)).cache().prefetch(AUTOTUNE)  

    if USE_TPU:
        tpu_strategy = load_tpu()

    #define loss, metrics, optimizer
    if USE_TPU:
        with tpu_strategy.scope():
            steps_per_epoch = processed_train.cardinality().numpy()
            optimizer = adam_w_optimizer(LEARNING_RATE, EPOCHS, steps_per_epoch)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = tf.metrics.BinaryAccuracy()

            model = models[MODEL]
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            history = model.fit(x=processed_train,
                                validation_data=processed_val,
                                steps_per_epoch=steps_per_epoch,
                                epochs=EPOCHS)
    else:
        steps_per_epoch = processed_train.cardinality().numpy()
        optimizer = adam_w_optimizer(LEARNING_RATE, EPOCHS, steps_per_epoch)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()

        model = models[MODEL]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        history = model.fit(x=processed_train,
                            validation_data=processed_val,
                            steps_per_epoch=steps_per_epoch,
                            epochs=EPOCHS)

    #save model
    #log metrics    

    return history

if __name__ =="__main__":
    main()

