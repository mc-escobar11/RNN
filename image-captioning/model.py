import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # pretrained model resnet50
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # replace the classifier with a fully connected embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, 
                            self.hidden_size, 
                            self.num_layers, 
                            batch_first=True)
        #For GRU Inicialization
        #for name, param in self.rnn.named_parameters():
        #       if 'bias' in name:
        #            nn.init.constant_(param, 0.0)
        #       elif 'weight_ih' in name:
        #            nn.init.kaiming_normal_(param)
        #       elif 'weight_hh' in name:
        #            nn.init.orthogonal_(param)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
    
    def forward(self, features, captions):
        
        caption_embed = self.word_embedding(captions[:, :-1])
        caption_embed = torch.cat((features.unsqueeze(dim=1), caption_embed),1)
        output, self.hidden = self.rnn(caption_embed)
        output = self.fc(output)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        """Samples captions for given image features."""
        output = []
        (h, c) = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device), torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        for i in range(max_len):
            x, (h, c) = self.rnn(inputs, (h, c))
            #x, h = self.rnn(inputs, h)
            x = self.fc(x)
            x = x.squeeze(1)
            predict = x.argmax(dim=1)
            if predict.item() == 1:
                break
            output.append(predict.item())
            inputs = self.word_embedding(predict.unsqueeze(0))
        return output