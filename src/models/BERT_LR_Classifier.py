import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

class BERT_LR_Preprocessor(tf.keras.Model):
    def __init__(self, added_layers, seq_length, preprocess_handle):
        super().__init__(name='preprocess')
        self.added_layers = added_layers
        self.preprocess_handle = preprocess_handle
        self.seq_length = seq_length
    
    def call(self, words):
        preprocess_model = hub.load(self.preprocess_handle) 
        tokenizer = hub.KerasLayer(preprocess_model.tokenize, name='tokenizer')
        packer = hub.KerasLayer(preprocess_model.bert_pack_inputs,
                          arguments=dict(seq_length=self.seq_length), name='packer')

        input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        x = [tokenizer(input)]
        output = packer(x)
        return tf.keras.Model(input, output)(words)

class BERT_LR_Classifier(tf.keras.Model):
    def __init__(self, encoder_handle):
        super().__init__(name='prediction')
        self.encoder = hub.KerasLayer(encoder_handle, trainable=True)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, preprocessed_text):
        encoder_outputs = self.encoder(preprocessed_text)
        pooled_output = encoder_outputs['pooled_output']
        x = self.dropout(pooled_output)
        x = self.dense(x)
        return x
