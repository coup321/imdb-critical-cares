from msilib.schema import Class
import tensorflow as tf
import tensorflow_hub as hub

class BERT_LR_Preprocesser(tf.keras.Model):
    def __init__(self, layers, seq_length, handle):
        super().__init__(name='preprocess')
        self.layers = layers
        self.handle=handle
        self.seq_length = seq_length

        self.preprocess_model = hub.load(self.handle) 
        self.tokenizer = hub.KerasLayer(self.preprocess_model.tokenize, name='tokenizer')
        self.packer = hub.KerasLayer(self.preprocess_model.bert_pack_inputs,
                          arguments=dict(seq_length=self.seq_length), name='packer')
    
    def call(self, words):
        return tf.Sequential(layers = self.layers + [self.tokenizer, self.packer])(words)
     
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