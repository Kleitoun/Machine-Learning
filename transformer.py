import torch 
import math
import torch.nn as nn
from torchtext.datautils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

## Dataset Preparation and Tokenization 

# Example dataset
texts = ["What is the capital of France?", "The capital of France is Paris.", "Who is the president of the USA?", "The president of the USA is Joe Biden."]

# Tokenization
tokenizer = get_tokenizer('basic_english')

#Build vocabulary
def yield_tokens(texts):
  for text in texts:
    yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(text), specials=["<unk>","<pad>","<eos>","<bos>"])
vocab.set_default_index(vocab["unk"])

tokens = [vocab[token] for token in tokenizer(text)]

## Positional Encoding

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=10000):
    super(PositionalEncoding, self).__init__
    self.pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
    self.pe[:,0::2] = torch.sin(position * div_term)
    self.pe[:,1::2] = torch.cos(position * div_term)
    self.pe = self.pe.unsequeeze(0)
  
  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return x
    
## Embedding Layer

class EmbeddingLayer(nn.Module):
  def __init__(self, vocab_size, d_model):
    super(EmbeddingLayer, self).__init__
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.embedding.embedding_dim)

## Self Attention (can handle multi head as well)

class Attention(nn.Module):
  def __init__(self, d_model, num_heads=1, dropout=0.1):
    super(Attention, self).__init__
    assert d_model % num_heads == 0
    self.d_k = d_model // num_heads
    self.num_heads = num_heads

    self.q_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.out = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, mask=None):
    batch_size = q.size(0)

    q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
    k = self.q_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
    v = self.q_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)

    scores = torch.matmul(q, k.transpose(-2, -1)/ math.sqrt(self.d_k))
    attention = nn.functional.softmax(scores, dim=-1)
    attention = self.dropout(attention)
    output = torch.matmul(attention, v)

    output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
    output = self.out(output)
    return output

## FFN 
class NeuralNetwork(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(NeuralNetwork, self).__init__
    self.linear1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(d_ff, d_model)

  def forward(self, x):
    x = self.dropout(nn.functional.relu(self.linear1))
    x = self.linear2(x)
    return x

## Encoder
class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super(EncoderLayer, self).__init__
    self.self_attention = Attention(d_model, num_heads, dropout)
    self.ff = NeuralNetwork(d_model, d_ff, dropout)
    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    z = self.self_attention(x, x, x)
    x = x + self.dropout(z)
    x = self.layernorm1(x)

    z = self.ff(x)
    x = x + self.dropout(x)
    x = self.layernorm2(x)
    
    return x

class Encoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=10000, dropout=0.1):
    super(Encoder,self).__init__
    self.embedding = EmbeddingLayer(vocab_size, d_model)
    self.pos_encoder = PositionalEncoding(d_model, max_len)
    self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)] for _ in range(num_layers))
    self.dropout = nn.Dropout(dropout)
    self.d_model = d_model 

  def forward(self, x):
    x = self.embedding(x)*math.sqrt(self.d_model)
    x = self.pos_encoder(x)
    x = self.dropout(x)

    for layer in self.layers:
      x = layer(x)

    return x

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super(DecoderLayer, self).__init__
    self.self_attention = Attention(d_model, num_heads, dropout)
    self.cross_attention = Attention(d_model, num_heads, dropout)
    self.ff = NeuralNetwork(d_model, d_ff, dropout) 
    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.layernorm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, z):
    s_a = self.self_attention(x, x, x)
    x = x + self.dropout(s_a)
    x = self.layernorm1(x)

    c_a = self.cross_attention(x, z, z)
    x = x + self.dropout(c_a)
    x = self.layernorm2(x)

    target = self.ff(x)
    x = x + self.dropout(target)
    x = self.layernorm3(x)

    return x

class Decoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=10000, dropout=0.1):
    super(Decoder, self).__init__
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoder = PositionalEncoding(d_model, max_len)
    self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    self.dropout = nn.Dropout(nn)
    self.d_model = d_model
    self.output = nn.Linear(d_model, vocab_size)

  def forward(self, x, z): 
    x = self.embedding(x) * math.sqrt(self.d_model)
    x = self.pos_encoder(x)
    x = self.dropout(x)

    for layers in self.layers:
      x = layer(x, z)

    output = self.output(x)
    return output

class Transformer(nn.Module):
  def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff, vocab_size, max_len=10000, dropout=0.1):
    super(Transformer, self).__init__
    self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, vocab_size, max_len=10000, dropout=0.1)
    self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, vocab_size, max_len=10000, dropout=0.1)

  def forward(self, x, z):
    input = self.encoder(x)
    output = self.decoder(z, x)
    return output


    


    
                         




