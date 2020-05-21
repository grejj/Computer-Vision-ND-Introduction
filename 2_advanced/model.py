import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        # set class variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        # lstm
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # linear layer for converting lstm output to word
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # pass captions through embedding layer
        captions = self.embed(captions[:,:-1])
        # put features and captions together into LSTM
        output, hidden = self.lstm(torch.cat((features.unsqueeze(1), captions), 1)) 
        # convert to words
        output = self.linear(output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
          
        sentence = []
    
        for _ in range(max_len):
            
            # pass through LSTM
            output, states = self.lstm(inputs, states)
            # pass LSTM output through linear to convert to words
            output = self.linear(output)
            # get most likely word
            prob, word = output.max(2)
            # add word to sentence
            sentence.append(word.item())
            # pass output word into next iteration
            inputs = self.embed(word)
    
        return sentence