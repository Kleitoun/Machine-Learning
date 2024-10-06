import torch

sentence = 'Attention is all you need'

dc = {s: i for i, s in enumerate(sorted(sentence.replace(',','').split()))}

s = torch.tensor([dc[s] for s in sentence.replace(',','').split()])
sentence_len = s.size[0]
embedding_size = 40

embed = torch.nn.Embedding(sentence_len, embedding_size)

embedded_sentence = embed(s).detach()
embedded_sentence_2 = torch.rand(8,embedding_size)

torch.manual_seed(42)

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 20, 20, 16

W_q = torch.nn.Parameters(torch.rand(d_q, d))
W_k = torch.nn.Parameters(torch.rand(d_k, d))
W_v = torch.nn.Parameters(torch.rand(d_v, d))

#self attention
queries = W_q.matmul(embedded_sentence.T).T
keys = W_k.matmul(embedded_sentence.T).T 
values = W_v.matmul(embedded_sentence.T).T

interaction = (queries @ keys.T)/d_k**0.5
attention_weights = torch.nn.functional.softmax(interaction, dim=0)
output = attention_weights.matmul(values)

#cross attention
keys_2 = W_k.matmul(embedded_sentence_2.T).T
values_2 = W_v.matmul(embedded_sentence_2.T).T

cross_interaction = (queries.keys_2.T)/d_k**0.5
cross_attn = torch.nn.functional.softmax(cross_interaction, dim=0)
cross_out = cross_attn.matmul(values_2)





