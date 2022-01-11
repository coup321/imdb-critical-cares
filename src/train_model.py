import tensorflow as tf
from official.nlp import optimization
from src.models.BERT_LR_Classifier import BERT_LR_Classifier
import datetime

class Trainer:
    def __init__(self, train, val, model, tpu_strategy=None):
        self.train = train
        self.val = val
        self.model = model
        self.tpu_strategy = tpu_strategy
        self.learning_rate = None
        self.epochs = None

    def compile_model(self, learning_rate, epochs):
        self.set_learning_rate(learning_rate)
        self.set_epochs(epochs)
        optimizer = self._adam_w_optimizer(self.learning_rate, self.epochs)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    def train_model(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = self.model.fit(
            x=self.train,
            validation_data=self.val,
            steps_per_epoch=self.train.cardinality().numpy(),
            epochs=self.epochs,
            callbacks=[tensorboard_callback])
            

        return history

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_epochs(self, epochs):
        self.epochs = epochs

    def _adam_w_optimizer(self, learning_rate, epochs):
        steps_per_epoch = self.train.cardinality().numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = num_train_steps // 10

        optimizer = optimization.create_optimizer(
            init_lr=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw'
            )
        return optimizer


def load_tpu():
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='') #get cluster
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu) #connect to cluster
    tf.tpu.experimental.initialize_tpu_system(tpu) #initialize cluster
    tpu_strategy = tf.distribute.TPUStrategy(tpu)  #define TPU strategy
    print(f'Number of TPU workers: ', tpu_strategy.num_replicas_in_sync)
    return tpu_strategy

def train_model(train, val, model_handle, learning_rate, epochs):
    model = BERT_LR_Classifier(model_handle)
    trainer = Trainer(train, val, model)
    trainer.compile_model(learning_rate, epochs)
    history = trainer.train_model()
    return history
        