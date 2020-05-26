# This is a NMT of english natural language to Sparql

#Reading the data files
#%matplotlib inline

import re, string
import numpy as np
import spacy
from spacy.symbols import ORTH
ent_tok = spacy.load('en')

# Incase of maximum recursion exceeded 
#import sys
#sys.setrecursionlimit(150000)

def en_tokenizer(text):
	text = text.lower()
	return [t.text for t in ent_tok.tokenizer(text)]

eng = open('data.en', 'r').readlines()
sparql = open('data.sparql', 'r').readlines()

# to see the contents of NL and sparql datasets
#print(eng[:2], sparql[:2])

token = []
token_s = []

def make_corpus_en(corpus):
	for line in corpus:
		if '\n' in line:
			new_line = line.replace('\n', '')
			token.append(new_line)
	return token
	
def make_corpus_spql(corpus):
	for line in corpus:
		if '\n' in line:
			new_line = line.replace('\n', '')
			token_s.append(new_line)
	return token_s	
	
##make corpus
	
en = make_corpus_en(eng)
spql = make_corpus_spql(sparql)	

##create tokens
en_toks = [en_tokenizer(text) for text in en]
spql_toks = [en_tokenizer(text) for text in spql]

#print(en_tokenizer(en[0])) 
#print(en_tokenizer(spql[0]))
##printing tokens
#print(en_toks[:2], spql_toks[:2])
#print(len(en_toks), len(spql_toks))
#print(en_toks[:2]), print(spql_toks[:2])
#print(type(en_toks), type(spql_toks))

## Creating a dictionary 
from collections import Counter, defaultdict

def numericalize_tok(tokens, max_vocab=50000, min_freq=0, unk_tok="xxunk", pad_tok="xxpad", bos_tok="xxbos", eos_tok="xxeos"):
	if isinstance(tokens, str):
		raise ValueError("Expected to recieve a list of tokens. Recieved a string instead")
	if isinstance(tokens[0], list):
		tokens = [p for o in tokens for p in o]
	freq = Counter(tokens)
	int2tok = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
	unk_id = 3
	int2tok.insert(0, bos_tok)
	int2tok.insert(1, pad_tok)
	int2tok.insert(2, eos_tok)
	int2tok.insert(unk_id, unk_tok)
	tok2int = defaultdict(lambda:unk_id, {v:k for k,v in enumerate(int2tok)})
	return int2tok, tok2int
	
#int2en, en2int = numericalize_tok(en_toks)
#int2spql, spql2int = numericalize_tok(spql_toks)

#print(len(int2en), len(int2spql))
#print(int2en[:2]), print(int2spql[:2])
#print(type(int2en), type(int2spql))
#saving tokens
import pickle

#save list of tokens
#int2en_out = open("int2en.pkl","wb")
#pickle.dump(int2en, int2en_out)
#int2en_out.close()

#int2spql_out = open("int2spql.pkl", "wb")
#pickle.dump(int2spql, int2spql_out)
#int2spql_out.close()

# loading tokens
int2en_in = open("int2en.pkl","rb")
int2en = pickle.load(int2en_in)

int2spql_in = open("int2spql.pkl","rb")
int2spql = pickle.load(int2spql_in)
print(type(int2en), int2en[20:30])
print(type(int2spql), int2spql[20:30])

##Create indices
en2int = defaultdict(lambda:3, {v:k for k, v in enumerate(int2en)})
spql2int = defaultdict(lambda:3, {v:k for k, v in enumerate(int2spql)})

print(len(int2en), len(int2spql))
eng_ids = np.array([[0]+[en2int[o] for o in sent]+[2] for sent in en_toks])
spql_ids = np.array([[0]+[spql2int[o] for o in sent]+[2] for sent in spql_toks])
print(len(eng_ids), len(spql_ids), eng_ids[10], spql_ids[10])

##Splitting train and evaluation set
np.random.seed(42)

trn_keep = np.random.rand(len(eng_ids))>0.1
eng_trn, spql_trn = eng_ids[trn_keep], spql_ids[trn_keep]
eng_val, spql_val = eng_ids[~trn_keep], spql_ids[~trn_keep]
print(len(eng_trn), len(spql_trn))
print(len(eng_val), len(spql_val))

#Using Pytorch framework

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
#Creating a sequence2sequence

from numpy import array as A 
class Seq2SeqDataset(Dataset):
    def __init__(self, x, y): self.x,self.y = x,y
    def __getitem__(self, idx): return A(self.x[idx]), A(self.y[idx])
    def __len__(self): return len(self.x)

#Padding for batch normalization
import keras
from keras.preprocessing.sequence import pad_sequences
englen_90 = int(np.percentile([len(o) for o in eng_ids], 99))
spqllen_90 = int(np.percentile([len(o) for o in spql_ids], 99))
print(englen_90, spqllen_90)

eng_ids = pad_sequences(eng_ids, maxlen=24, dtype='int32', padding='post', truncating='post', value=1)
spql_ids = pad_sequences(spql_ids, maxlen=26, dtype='int32', padding='post', truncating='post', value=1)
trn_keep = np.random.rand(len(eng_ids))>0.1
eng_trn, spql_trn = eng_ids[trn_keep], spql_ids[trn_keep]
eng_val, spql_val = eng_ids[~trn_keep], spql_ids[~trn_keep]
#print(len(eng_trn), len(spql_trn))
#print(len(eng_val), len(spql_val))

#training sequences	
trn_ds = Seq2SeqDataset(eng_trn, spql_trn)
val_ds = Seq2SeqDataset(eng_val, spql_val)

bs = 120 # batch size

#Loading training Dataset
trn_dl = DataLoader(trn_ds, batch_size = bs, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=bs)

x, y = next(iter(val_dl))
print(x.size(), y.size())

# Loading fasttext vectors
# https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md

import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    header = fin.readline().split()
    n, d = int(header[0]), int(header[1])
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype=float)
    return data, int(n), int(d)
    
# get word vectors
#wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
#wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
#unzip wiki-news-300d-1M.vec.zip
#gunzip cc.ja.300.vec.gz
#mv wiki-news-300d-1M.vec data/
#mv cc.ja.300.vec data/
 
#Loading vectors 
eng_vecs,_,dim_eng_vec = load_vectors('wiki-news-300d-1M.vec')
spql_vecs,_,dim_spql_vec = load_vectors('wiki-news-300d-1M.vec')
   
#Creating embeddings
def create_emb(vecs, itos, em_sz):
	emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
	if vecs is None: return emb
	wgts = emb.weight.data
	miss = []
	for i,w in enumerate(itos):
		try: wgts[i] = torch.from_numpy(vecs[w])
		except: miss.append(w)
	print('Number of unknowns in data: {}'.format(len(miss)))
	return emb
	
def V(tensor, reg_grad=True):
	if torch.cuda.is_available():return Variable(tensor.cuda())
	else: return Variable(tensor)

##Seq2Seq Architecture
#...a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed 
#dimensionality, and then another deep LSTM to decode the target sequence from the vector.

#Sutskever et al., 2014
#nl:number of layers
class Seq2Seq(nn.Module):
	def __init__(self, int2en,int2spql,em_sz,spql_vecs=None,eng_vecs=None,nh=128,out_sl=24,dropf=1,nl=2):
		super().__init__()
		#encoder
		self.nl,self.nh,self.em_sz,self.out_sl = nl,nh,em_sz,out_sl
		self.emb_enc = create_emb(eng_vecs,int2en,em_sz)
		self.emb_drop = nn.Dropout(0.15*dropf)
		self.encoder = nn.GRU(em_sz,nh,num_layers=nl,dropout=0.25*dropf, bidirectional=True)
		#decoder
		self.emb_dec = create_emb(spql_vecs,int2spql,em_sz)
		self.decoder = nn.GRU(em_sz,nh*2,num_layers=nl,dropout=0.25*dropf)
		self.out_drop = nn.Dropout(0.35*dropf)
		self.out = nn.Linear(nh*2,len(int2spql))
	
	def forward(self,inp,y=None):
		sl, bs = inp.size()
		emb_in = self.emb_drop(self.emb_enc(inp))
		h_n = self.initHidden(bs)
		enc_out, h_n = self.encoder(emb_in,h_n)
		h_n = h_n.view(2,2,bs,-1).permute(0,2,1,3).contiguous().view(self.nl,bs,-1)
		
		dec_inp = V(torch.zeros(bs).long())
		res = []
		for i in range(self.out_sl):
			dec_emb = self.emb_dec(dec_inp)
			outp,h_n = self.decoder(dec_emb.unsqueeze(0),h_n)
			outp = F.log_softmax(self.out(self.out_drop(outp[0])),dim=-1)
			res.append(outp)
			dec_inp = outp.data.max(1)[1]
			if (dec_inp==1).all(): break
		return torch.stack(res)
		
	def initHidden(self,bs):
		return V(torch.zeros([self.nl*2,bs,self.nh]))
    	
#Why Bidirectional Encoder
#Finally, we found that reversing the order of the words in all source sentences (butnot target sentences) 
#improved the LSTM’s performance markedly, because doing so introduced many short term dependencies between 
#the source and the targetsentence which made the optimization problem easier.

#Sutskever et al., 2014

#Input and output sequences may not directly map to each other so preserving information from both passes
# of the input sequence will help learn how tokens relate to each other. For example in a translation task, 
#subject and object can be in opposite positions depending on the language structure.    
seq2seq = Seq2Seq(int2en, int2spql,300,eng_vecs=None,spql_vecs=None)
#print(seq2seq.cuda())	
print(seq2seq)	

out = seq2seq(V(x.transpose(1,0).long()))
print(out.size())

#Loss Function: We will use Cross Entropy Loss as we are trying to classify ot the correct words. 
#Cross entropy loss can be simplified to:
#cross_entropy = sum(-log(y_pred) for y_pred in y_preds)
#where y_pred is the likelihood of the target class predicted by the model. This is a good 
#loss function for classification because if the likelihood of the correct class is low, the loss 
#value goes up and if it is high, the loss value goes down.

def seq2seq_loss(input, target):
    sl,bs = target.size()
    sl_in,bs_in,nc = input.size()
    if sl>sl_in: input = F.pad(input, (0,0,0,0,0,sl-sl_in))
    input = input[:sl]
    return F.cross_entropy(input.view(-1,nc), target.view(-1))
    
print(seq2seq_loss(out,V(y.transpose(1,0).long())))

def step(x, y, epoch, m, crit, opt, clip=None):
    output = m(x, y)
    if isinstance(output,tuple): output = output[0]
    opt.zero_grad()
    loss = crit(output, y)
    loss.backward()
    if clip:
        nn.utils.clip_grad_norm_(m.parameters(), clip)
    opt.step()
    return loss.data.item()
    
from tqdm import tqdm


def train(trn_dl,val_dl,model,crit,opt,epochs=5,clip=None):
    for epoch in range(epochs):
        loss_val = loss_trn = 0
        with tqdm(total=len(trn_dl)) as pbar:
            model.train()
            for i, ds in enumerate(trn_dl):
                x, y = ds
                #if isinstance(x,tuple): x = x[0]
                x, y = x.transpose(1,0), y.transpose(1,0)
                loss = step(V(x.long()),V(y.long()),epoch,model,crit,opt)
                loss_trn += loss
                pbar.update()
        model.eval()
        for i, ds in enumerate(val_dl):
            with torch.no_grad():
                x, y = ds
                #if isinstance(x,tuple): x = x[0]
                x, y = x.transpose(1,0), y.transpose(1,0)
                out = model(V(x.long()))
                if isinstance(out,tuple): out = out[0]
                loss_val+= crit(out, V(y.long()))
                #loss_val +=loss
        print(f'Epoch: {epoch} trn loss: {loss_trn/len(trn_dl)} val loss: {loss_val/len(val_dl)}')
        
from torch import optim
opt = optim.Adam(seq2seq.parameters(),lr=3e-3,betas=(0.7,0.8))
train(trn_dl,val_dl,seq2seq,seq2seq_loss,opt,epochs=5)

#Print trained output

def produce_out(val_dl, model,int2en,int2spql,interval=(20,30)):
    model.eval()
    x,y = next(iter(val_dl))
    x, y = x.transpose(1,0), y.transpose(1,0)
    probs = seq2seq(V(x.long()))
    if isinstance(probs,tuple): probs = probs[0] 
    preds = A(probs.max(2)[1].cpu())
    for i in range(interval[0],interval[1]):
        print(' '.join([int2en[o] for o in x[:,i] if o not in [0,1,2]]))
        print(''.join([int2spql[o] for o in y[:,i] if o not in [0,1,2]]))
        print(''.join([int2spql[o] for o in preds[:,i] if o not in [0,1,2]]))
        print()


