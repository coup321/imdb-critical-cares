from src.train_model import adam_w_optimizer
from models.BERT_LR_Classifier import BERT_LR_Preprocesser, BERT_LR_Classifier
import tensorflow as tf

DATA_DIR = "./data/raw"
LOG_DIR = "./logs"
EPOCHS = 5
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
PREPROCESSER = 'BERT_LR_Classifier'
MODEL = 'BERT_LR_Classifier'

models = {
    'BERT_LR_Classifier' : BERT_LR_Classifier()
}

def main(preprocesser=PREPROCESSER, model=MODEL, tpu=False, val=True, test=False):
    #load data
    data = [0,0,0]

    #preprocess data
    data = [preprocesser(dataset) for dataset in data]

    #define loss, metrics, optimizer
    steps_per_epoch = train.cardinality().numpy()
    optimizer = adam_w_optimizer(LEARNING_RATE, EPOCHS, steps_per_epoch)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(x=train,
                        validation_data=val,
                        steps_per_epoch=steps_per_epoch,
                        epochs=EPOCHS)

    #save model
    #log metrics    

    pass

if __name__ =="__main__":
    main()