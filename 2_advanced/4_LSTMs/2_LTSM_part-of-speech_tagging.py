#!/usr/bin/env python3

# USE LSTM TO PREDICT PARTS OF SPEECH

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import matplotlib.pyplot as plt 
import numpy as np

training_data = [
    ("The cat ate the cheese".lower().split(), ["DET", "NN", "V", "DET", "NN"]),
    ("She read that book".lower().split(), ["NN", "V", "DET", "NN"]),
    ("The dog loves art".lower().split(), ["DET", "NN", "V", "NN"]),
    ("The elephant answers the phone".lower().split(), ["DET", "NN", "V", "DET", "NN"])
]

# create dictionary to map words to indices
word2idx = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
        
# create dictionary that maps tags to indices
tag2idx = {"DET": 0, "NN": 1, "V": 2}
print(word2idx)

# helper function, convert sequence of words to Tensor of numerical values
def prepare_sequence(seq, to_idx):
    idxs = [to_idx[w] for w in seq]
    idxs = np.array(idxs)
    return torch.from_numpy(idxs)

example_input = prepare_sequence("The dog answers the phone".lower().split(), word2idx)
print(example_input)

# create model -> model assumes input broken into sequence of words, words come from larger list/vocabulary, limited set of tags to predict for each word

# pass LSTM over sentence and apply softmax to hidden state of LSTM, result is vector of tag scores 
class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        ''' Initialize layers '''
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        # embedding layer that turns words into a vector of specified size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # the LSTM takes embedded word vectors as inputs and outputs hidden states
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # linear layer mapping hidden state to output dimension
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # initialize hidden state
        self.hidden = self.init_hidden()

    def init_hidden(self):
        ''' First hidden state, will be zero'''
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
    
    def forward(self, sentence):
        ''' Define feedforward behavior of model '''
        # create embedded word vector for each word in sentence
        embeds = self.word_embeddings(sentence)

        # get output and hidden state from lstm by inputting embeddings and previous hidden state
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)

        # get scores for most likely tag for a word
        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_outputs, dim=1)

        return tag_scores

# define training of model

# embedding dimension defines size of word vectors, kept small for small training set
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(tag2idx))

# define loss to be NLLLoss because output tag scores uses softmax
loss_function = nn.NLLLoss()
# define optimizer as standard, basic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)

# test model without training
test_sentence = "The cheese loves the elephant".lower().split()
# element [i,j] of output is score for tag j for word i
inputs = prepare_sequence(test_sentence, word2idx)
tag_scores = model(inputs)
print(tag_scores)

# tag_scores outputs a vector of tag scores for each word in an inpit sentence
# to get the most likely tag index, we grab the index with the maximum score!
# recall that these numbers correspond to tag2idx = {"DET": 0, "NN": 1, "V": 2}
_, predicted_tags = torch.max(tag_scores, 1)
print('\n')
print('Predicted tags: \n',predicted_tags)

# train model
n_epochs = 300

for epoch in range(n_epochs):
    epoch_loss = 0.0

    # get all sentences and corresponding tags in training data
    for sentence, tags in training_data:
        # zero gradients
        model.zero_grad()
        # zero hidden state of LSTM, detaches it from history
        model.hidden = model.init_hidden()
        # prepare inputs for processing by network, setences -> Tensors with numerical indices
        sentence_in = prepare_sequence(sentence, word2idx)
        targets = prepare_sequence(tags, tag2idx)
        # forward pass to get tag scores
        tag_scores = model(sentence_in)
        # get loss and gradients
        loss = loss_function(tag_scores, targets)
        epoch_loss += loss.item()
        loss.backward()
        # update model parameters
        optimizer.step()

    # print out average loss every 20 epochs
    if (epoch % 20 == 19):
        print("Epoch: {}, loss: {}".format(epoch+1, epoch_loss/len(training_data)))

# test model after training
test_sentence = "The cheese loves the elephant".lower().split()
# element [i,j] of output is score for tag j for word i
inputs = prepare_sequence(test_sentence, word2idx)
tag_scores = model(inputs)
print(tag_scores)

# tag_scores outputs a vector of tag scores for each word in an inpit sentence
# to get the most likely tag index, we grab the index with the maximum score!
# recall that these numbers correspond to tag2idx = {"DET": 0, "NN": 1, "V": 2}
_, predicted_tags = torch.max(tag_scores, 1)
print('\n')
print('Predicted tags: \n',predicted_tags)