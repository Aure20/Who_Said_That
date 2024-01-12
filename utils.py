import json 
import tensorflow_hub as hub 
import pandas as pd
from tqdm import tqdm 
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import requests
import time
import random
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import matplotlib.pyplot as plt
from quotes import Quotes
import seaborn as sns

def create_embeddings(json_path: str):
    """Given the quotes in json format saves a dataframe containing their embeddings'

    Args:
        json_path (str): filepath where the json is stored.
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = file.read()
    data_dict = json.loads(json_data)
    module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4' 
    model = hub.load(module_url)
    df = {}
    for author,quotes in tqdm(data_dict.items(), desc='Crete Embeddings'):
        curr_embs = model(quotes) 
        df.update({(author,quote):emb.numpy() for quote,emb in zip(quotes,curr_embs)})
    df = pd.DataFrame.from_dict(df, 'index', np.float16)
    df.to_csv('embeddings-300.csv')


def get_nn_model(df_path: str):
    """Given a dataframe where the first column contains the quote, return the Nearest neighbour. 

    Args:
        df_path (str): Dataframe Path
    """
    df = pd.read_csv(df_path)
    neigh = NearestNeighbors(n_jobs=6)
    neigh.fit(df.drop(columns=df.columns[0]))
    with open('full_nn.pkl', 'wb') as file:
        pickle.dump(neigh, file)

def get_weights(json_path: str):
    """Given the path of the json file containing the authors save a dataframe containing the authors and how many results they have.

    Args:
        json_path (str): path of json file
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = file.read()
    quotes = json.loads(json_data)
    author_weight = {}
    for i,author in enumerate(tqdm(list(quotes.keys())[11603:])): 
        link = f'https://www.googleapis.com/customsearch/v1?key=[KEY]=&q={author}&alt=json&fields=queries(request(totalResults))'
        f = requests.get(link)
        time.sleep(0.7)
        ngram = json.loads(f.text)
        try:
            total_results = int(ngram["queries"]["request"][0]["totalResults"])
        except KeyError:
            print(author)
            continue
        author_weight.update({author:total_results})
        if i%10 == 0:
            df = pd.DataFrame.from_dict(author_weight, 'index')
            df.to_csv('author-weights3.csv')

def plot_search_hit_counts(csv_path = 'test.csv'):
    quotes = Quotes('embeddings-300.csv', 'base_nn.pkl', 'https://tfhub.dev/google/universal-sentence-encoder/4')
    weighted_hits = 0
    base_hits = 0
    corrected_hits = 0

    test = pd.read_csv(csv_path)
    for index,row in test.iterrows(): 
        quote = row['Quote']
        quote = quote[len(quote)//4:int(len(quote)//4*3)]
        closest = quotes.get_nearest(row['Query'], mode='base')
        for result in closest:
            if quote in result[0]:
                base_hits += 1         
        closest = quotes.get_nearest(row['Query'], mode='weighted')
        for result in closest:
            if quote in result[0]:
                weighted_hits += 1
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            closest = quotes.get_nearest(row['Query'], mode='corrected')
            for result in closest:
                if quote in result[0]:
                    corrected_hits += 1

    categories = ['base', 'weighted', 'corrected']
    values = [base_hits, weighted_hits, corrected_hits]

    # Plotting
    plt.bar(categories, values, color=['blue', 'green', 'orange']) 
    plt.xlabel('Search Mechanism')
    plt.ylabel('Number in results') 
    plt.savefig('test_count.png')
    plt.show()

def parse_txt2csv(txt_path:str = "train.txt"):
    with open(txt_path, "r", encoding="utf-8") as file:
        quote_dict = {}
        current_quote = ''
        for line in file:
            if line[0:8] !='"quote":':
                if line == '\n':
                    continue
                quote_dict[line.rstrip('",\n').strip('"')] = current_quote
            else:
                current_quote = line.strip('"quote":" ""').rstrip('",""\n')

        data = pd.DataFrame(list(quote_dict.items()), columns=['Query', 'Quote'])
        data.to_csv('test.csv', index=False)



def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    g.plot()

def run_and_plot(messages_):
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    message_embeddings_ = embed(messages_)
    plot_similarity(messages_, message_embeddings_, 90)

