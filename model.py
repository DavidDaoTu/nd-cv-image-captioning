import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        # Using ResNet-34 for a lighter model
        resnet = models.resnet18(weights='DEFAULT')
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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, batch_size=32):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Hyper-parameters: fine tuning later
        self.num_layers = num_layers
        self.drop_prob = 0.5 
        self.batch_size = batch_size
        
        # Word embedding layer that turns a word into a vector with a specified size
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        # The LSTM layer takes embedded word vectors (of a specified size) as inputs
        # and output the hidden states (short-term memory) with size of hidden_size
        self.lstm = nn.LSTM(input_size=self.embed_size, 
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        
        # init hidden layer weights
        self.hidden_cell_state = self.init_hidden(self.batch_size)
        
        ### Dropout layer (as suggested by https://arxiv.org/pdf/1411.4555)
        self.dropout = nn.Dropout(self.drop_prob)
        
        ### The Linear layer that maps the hidden states (short-term memory) to
        # the vocab_size as output
        self.hidden2Vocab = nn.Linear(self.hidden_size, self.vocab_size)
        
        # init linear layer weights
        self.init_linear()      
        
        pass
    
    def init_linear(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.hidden2Vocab.bias.data.fill_(0)
        # FC weights as random uniform
        self.hidden2Vocab.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Setup the GPU device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # Randomly initialization for hidden state and cell state of LSTM
        hidden = (torch.rand(self.num_layers, batch_size, self.hidden_size).to(device),
                    torch.rand(self.num_layers, batch_size, self.hidden_size).to(device))
        return hidden
        
    def forward(self, features, captions):
        """forward phase returns output"""
        # Create embedded word vectors for captions [batch_size, seq_len]
        #   + Extract captions without the last stop word: <end> 
        #   + Create embedded word vectors with size [batch_size, seq_len, embed_size]
        # embeddings size: [batch_size, vocab_size, embed_size]
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        batch_size = captions.shape[0]
        seq_len = captions.shape[1]
        #print(f'batch_size ={batch_size}, seq_len = {seq_len}')
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            # init hidden layer weights
            self.hidden_cell_state = self.init_hidden(self.batch_size)
            
        # Reshape the image feature vector from [batch_size, embed_size]
        # to tensor size [batch_size, 1, embed_size]
        features = features.view(self.batch_size, 1, -1)
        
        # Concat the image feature tensor to the first position of embedded_word_vectors
        # inputs size [batch_size, seq_len, embed_size]
        inputs = torch.cat((features, embeddings), dim=1)
        
        ### Pass the inputs through the LSTM to get output
        # outputs size: [batch_size, seq_len, embed_size]
        outputs, self.hidden_cell_state = self.lstm(inputs, self.hidden_cell_state)
        
        ### Pass the output to the Dropout layer
        outputs = self.dropout(outputs)
        
        # Reshape the outputs to [batch_size * seq_len, hidden_size] required by Project
        outputs = outputs.reshape(batch_size * seq_len, self.hidden_size)
        
        ### Pass the LSTM output through Linear to get the next word
        outputs = self.hidden2Vocab(outputs)
        
        ## Reshape the outputs to a new shape required by the project
        # outptus size: [batch_size, seq_len, vocab_size]
        outputs = outputs.view(batch_size, seq_len, -1)
        
        # Check that your decoder satisfies some requirements of the project! :D        
        assert type(outputs)==torch.Tensor, "Decoder output needs to be a PyTorch Tensor."
        assert (outputs.shape[0]==batch_size) & (outputs.shape[1]==captions.shape[1]) & (outputs.shape[2]==self.vocab_size), "The shape of the decoder output is incorrect."
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """ accepts pre-processed image tensor (inputs) and 
        returns predicted sentence (list of tensor ids of length max_len) """
        predicted_sentence = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.hidden2Vocab(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_sentence.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence