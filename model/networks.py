from models import EncoderBlock
import tensorflow as tf
from embedding import EmbeddingLayer

class BERT(tf.keras.Model):
  def __init__(self, hidden_size, head_num):
    super(BERT, self).__init__()
    self.embedding = EmbeddingLayer(hidden_size, 256)
    self.EncoderBlock = EncoderBlock(hidden_size, head_num, 0.1, 1e-12)

  def __call__(self, data):
    x = self.embedding(data)
    _, x = self.EncoderBlock(x[0])
    return x