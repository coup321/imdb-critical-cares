import tensorflow as tf
from official.nlp import optimization

class Trainer:

    def __init__(self, data, preprocesser, classifier, logger, tpu=False, val=True, test=False):
        self.data         = data
        self.preprocessor = preprocesser
        self.model        = classifier
        self.logger       = logger
        self.tpu          = tpu
        self.val          = val
        self.test         = test

    def tpu_setup():
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='') #get cluster
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

        tf.config.experimental_connect_to_cluster(tpu) #connect to cluster
        tf.tpu.experimental.initialize_tpu_system(tpu) #initialize cluster
        tpu_strategy = tf.distribute.TPUStrategy(tpu)  #define TPU strategy
        print(f'Number of TPU workers: ', tpu_strategy.num_replicas_in_sync)

    def run_preprocess(self):
        return (self.preprocessor(dataset) for dataset in self.data)

    def train_model(self):
        pass


def adam_w_optimizer(learning_rate, epochs, steps_per_epoch):
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = num_training_steps // 10

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
    print(f'Number of TPU workers: ', tpu_strategy.num_replicas_in_sync)
    return tf.distribute.TPUStrategy(tpu)  #define TPU strategy