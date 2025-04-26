import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import pickle

# Load word2idx
with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

# Clean text function
def clean_text(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F" 
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'', text)

    text = text.replace('#', ' ')
    text = text.replace('@', ' ')

    symbols = re.compile(r'[^A-Za-z0-9 ]')
    text = symbols.sub(r'', text)

    text = text.lower()

    return text

# Text to sequence function
def text_to_sequence(text, word2idx, maxlen=55):
    words = text.split()
    seq = [word2idx.get(word, 0) for word in words]
    if len(seq) > maxlen:
        seq = seq[:maxlen]
    else:
        seq = [0]*(maxlen - len(seq)) + seq
    return np.array(seq)

# Define the BiLSTM class
class BiLSTM(nn.Module):
    def __init__(self, weights_matrix, output_size, hidden_dim, hidden_dim2, n_layers, drop_prob=0.5):
        super(BiLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Embedding layer
        num_embeddings, embedding_dim = weights_matrix.size()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(weights_matrix)
        self.embedding.weight.requires_grad = False  # Freeze embedding layer

        # BiLSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, bidirectional=True, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_size)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        # Embedding
        embeds = self.embedding(x)

        # LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Stack up LSTM outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)

        # Dropout and fully connected layers
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        # Sigmoid activation
        sig_out = self.sigmoid(out)

        # Reshape to batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # Get last batch of labels

        return sig_out, hidden

    def init_hidden(self, batch_size, train_on_gpu=False):
        weight = next(self.parameters()).data

        layers = self.n_layers * 2  # Multiply by 2 for bidirectionality
        if train_on_gpu:
            hidden = (weight.new(layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(layers, batch_size, self.hidden_dim).zero_())
        return hidden

# Load the embedding weights matrix
weights_matrix = torch.tensor(np.load('weights_matrix.npy'))

# Instantiate the model
output_size = 1
hidden_dim = 128
hidden_dim2 = 64
n_layers = 2

net = BiLSTM(weights_matrix, output_size, hidden_dim, hidden_dim2, n_layers)

# Load the model's state_dict
net.load_state_dict(torch.load('state_dict.pt', map_location=torch.device('cpu')))
net.eval()

# Streamlit app
def main():
    st.title("Disaster Tweet Classifier")
    st.write("Enter a tweet to classify whether it's about a real disaster or not.")

    user_input = st.text_area("Enter Tweet Text:")

    if st.button("Classify"):
        if user_input:
            # Preprocess input
            clean_input = clean_text(user_input)
            seq = text_to_sequence(clean_input, word2idx)
            input_tensor = torch.from_numpy(seq).unsqueeze(0).type(torch.LongTensor)

            # Initialize hidden state
            h = net.init_hidden(1, train_on_gpu=False)
            h = tuple([each.data for each in h])

            # Make prediction
            with torch.no_grad():
                output, h = net(input_tensor, h)
                prob = output.item()
                pred = int(torch.round(output).item())

            # Display result
            if pred == 1:
                st.success(f"This tweet is about a **real disaster**. (Probability: {prob:.4f})")
            else:
                st.info(f"This tweet is **not about a real disaster**. (Probability: {prob:.4f})")
        else:
            st.warning("Please enter some text to classify.")

if __name__ == '__main__':
    main()
