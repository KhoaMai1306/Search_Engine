import numpy as np
import pandas as pd
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
import torch
import os
from tqdm import tqdm
import pickle

def get_word_embedding(method):
    """
    Retrieve the embedding for a given word based on the specified method.

    Parameters:
    -----------
    word : str
        The word for which you want to retrieve the embedding.
    method : str
        The embedding method to use. Options include: 'glove', 'word2vec', 'fasttext'.
    sentence : str, optional
        The sentence from which the word is extracted (required for BERT-based embeddings).

    Returns:
    --------
    An object capable of outputting the word vector

    """

    if "glove" in method.lower():

        if os.path.exists("glove_word_embeddings.pkl"):
            with open('glove_word_embeddings.pkl', 'rb') as embedding_file:
                word_embeddings = pickle.load(embedding_file)
            return word_embeddings
        else:

            #change the path accordingly
            glove_file_path = 'C:/Users/User/Desktop/COSC4397/glove.840B.300d.txt'

            word_embeddings = {}
            faults = {}
            # Open the GloVe file and read line by line
            with open(glove_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing GloVe embeddings", leave=True, mininterval=1, ascii=True):
                    try:
                        values = line.split()  # Split the line into words and numbers
                        word = values[0]  # The first value is the word
                        vector = np.asarray(values[1:], dtype='float32')  # The remaining values are the embedding vector
                        word_embeddings[word] = vector
                    except:
                        values = line.split()  # Split the line into words and numbers
                        faults[values[0]] = values[1:]

            # Save the word_embeddings dictionary to a file
            with open('glove_word_embeddings.pkl', 'wb') as embedding_file:
                pickle.dump(word_embeddings, embedding_file)
            
            return word_embeddings

    elif "word2vec" in method.lower(): 
        model = api.load('word2vec-google-news-300')
        return model

    elif "fasttext" in method.lower():
        model = api.load('fasttext-wiki-news-subwords-300')
        return model
    else:
        print("Not supported. Enter one of the following: glove, word2vec, fastext")



def get_bert_word_embedding(word, sentence):
    '''
    sentence : str
        The sentence from which the word is extracted (required for BERT-based embeddings).

    word : str
        The word in question
    '''

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Get the embeddings (outputs of the last hidden layer)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get token embeddings from the output
    token_embeddings = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

    # Get the tokens corresponding to the sentence
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Find the index of the word in the tokenized output
    word_index = tokens.index(word)

    # Extract the embedding for the word
    return token_embeddings[0, word_index, :].numpy()



def get_bert_text_embedding(sentence):

    """
    Generates a BERT-based sentence embedding by averaging the token embeddings.

    Parameters:
    -----------
    sentence : str
        The input sentence for which the embedding is required.

    Returns:
    --------
    torch.Tensor
        The sentence embedding as a tensor of shape [1, hidden_size].
    """

    # Load pre-trained BERT model and tokenizer (BERT base uncased)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encode the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    
    # Get the hidden states from the model
    with torch.no_grad():  # No need for gradients in this context
        outputs = model(**inputs)

    # Get the embeddings for the last hidden state
    last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
    
    # Get the sentence embedding by averaging the token embeddings (excluding [CLS] and [SEP] tokens)
    sentence_embedding = torch.mean(last_hidden_state, dim=1)  # Shape: [batch_size, hidden_size]
    
    return sentence_embedding