from src.train_model import load_tpu, train_model
from src.load_data import load_data
from src.models.BERT_LR_Classifier import BERT_LR_Preprocessor
import tensorflow as tf
import os
tf.get_logger().setLevel('ERROR') 



EPOCHS = 2
LEARNING_RATE = 5e-5
BATCH_SIZE = 32 * 8
TRAIN_SIZE = 0.6
TEST = False
LOG_DIR = "./logs"
PREPROCESSOR = 'BERT_LR_Preprocessor'
MODEL = 'BERT_LR_Classifier'
USE_TPU = True
AUTOTUNE = tf.data.AUTOTUNE

preprocessor_dict = {
    'BERT_LR_Preprocessor':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
}

models = {
    'BERT_LR_Classifier':'https://tfhub.dev/google/experts/bert/pubmed/2'
}

def main():
    if USE_TPU:
        os.environ["TFHUB_MODEL_LOAD_FORMAT"]="UNCOMPRESSED"
        tpu_strategy = load_tpu()
        
    #load data
    train, val, test = load_data(batch_size=BATCH_SIZE,train_size=TRAIN_SIZE)

    if TEST:
        train = test

    #process data
    preprocessor =  BERT_LR_Preprocessor([], 128, preprocessor_dict[PREPROCESSOR])
    processed_train = train.map(lambda x, y: (preprocessor(x), y)).cache().prefetch(AUTOTUNE)
    processed_val = val.map(lambda x, y: (preprocessor(x), y)).cache().prefetch(AUTOTUNE)  

    #define model and train
    if USE_TPU:
      with tpu_strategy.scope():
        train_model(processed_train, processed_val, models[MODEL], LEARNING_RATE, EPOCHS)
    else:
        train_model(processed_train, processed_val, models[MODEL], LEARNING_RATE, EPOCHS)

if __name__ =="__main__":
    main()

