import tensorflow_hub as hub 
import pandas as pd 
import numpy as np 
import pickle 
from model import Autoencoder
import torch

class Quotes():
    def __init__(self, authors_path, base_nn_path, module_path):
        self.quotes = pd.read_csv(authors_path, usecols=['Unnamed: 0']).to_numpy() #Change this in the future
        self.author_weights = pd.read_csv('author_weights.csv') 
        self.encoder_model = hub.load(module_path)
        with open(base_nn_path, 'rb') as file:
            self.base_nn_model = pickle.load(file)
        self.autoencoder = Autoencoder() 
        self.autoencoder.load_state_dict(torch.load('autoencoder_model.pth'))
        self.autoencoder.eval()

    def get_nearest(self,quote: str, mode: str = 'base')->np.ndarray:
        #Get embedding of the quote we are searching for
        quote_embedding = self.encoder_model([quote])

        if mode == 'base' or mode == 'weighted':
            distances, indices = self.base_nn_model.kneighbors(quote_embedding.numpy(), n_neighbors=20) #Add number of neighbours
            nearest_quotes = self.quotes[indices[0]]

            if mode == 'weighted':
                nearest_authors = [nearest_quotes[i][0].split(',')[0].strip('(\'').rstrip('\'').strip('\"').rstrip('\"') for i in range(len(nearest_quotes))]
                corrected_weights = []
                for distance,author in zip(distances[0],nearest_authors):
                    author_weight = self.author_weights.loc[self.author_weights['authors'] == author]['weights'].max()
                    corrected_weights.append(author_weight/(distance*10))
                sorted_indices = np.flip(np.argsort(corrected_weights))
                nearest_quotes = self.quotes[indices[0][sorted_indices]]

        if mode == 'exact': 
            #Weird format to be consistent
            nearest_quotes = np.array([[current_quote[0]] for current_quote in self.quotes if quote.lower() in current_quote[0].lower()])
        
        if mode == 'corrected':
            quote_embedding = torch.tensor(quote_embedding.numpy(), dtype=torch.float32)
            quote_embedding = self.autoencoder(quote_embedding)
            distances, indices = self.base_nn_model.kneighbors(quote_embedding.detach().numpy(), n_neighbors=20)
            nearest_quotes = self.quotes[indices[0]]
            nearest_authors = [nearest_quotes[i][0].split(',')[0].strip('(\'').rstrip('\'').strip('\"').rstrip('\"') for i in range(len(nearest_quotes))]
            corrected_weights = []
            for distance,author in zip(distances[0],nearest_authors):
                author_weight = self.author_weights.loc[self.author_weights['authors'] == author]['weights'].max()
                corrected_weights.append(author_weight/(distance*10))
                sorted_indices = np.flip(np.argsort(corrected_weights))
                nearest_quotes = self.quotes[indices[0][sorted_indices]]

        return nearest_quotes
