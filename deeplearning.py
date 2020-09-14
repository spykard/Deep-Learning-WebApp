# -*- coding: utf-8 -*-
"""Deep_Learning.ipynb

## Base Code

### Prerequisites

Python 3.6+ is required because of certain performance optimization steps, such as the use of f-strings.  
GPU usage is required, e.g. the LSTM layers are specific CuDNN ones.
"""

# Commented out IPython magic to ensure Python compatibility.
### Parameters ###
# %tensorflow_version 1.x  # Set the tensorflow version
use_google_drive = True #@param {type:"boolean"}
gpu_fraction_usage = 0.9 #@param {type:"slider", min:0, max:1, step:0.1}
###            ###

# GPU usage settings for Tensorflow backend
from tensorflow import ConfigProto, Session
from keras import backend as K
if K.backend() == "tensorflow":
    from keras.backend.tensorflow_backend import set_session
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction_usage
    set_session(Session(config=config))
else:
    raise ValueError("Keras is not using the 'tensorflow' backend")

"""### Preprocessing"""

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, CuDNNLSTM, Bidirectional, Dropout 
from keras.layers import SimpleRNN, GRU, GlobalAveragePooling1D, Conv1D, Flatten
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD, Adagrad
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model, np_utils
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
###            ###
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
###            ###
import random
from tensorflow import set_random_seed

### Parameters ###
random_state = 22 #@param {type:"slider", min:0, max:100, step:1}
dataset_name = "IMDb Large Movie Review Dataset" #@param ["IMDb Large Movie Review Dataset", "Movie Review Subjectivity Dataset", "Movie Review Polarity Dataset", "SCH Dataset", "Finegrained Sentiment Dataset"]
feature_count = 15000 #@param {type:"slider", min:0, max:140000, step:5000}
###            ###

# Reproducibility
random.seed(random_state)  # Python's seed
np.random.seed(random_state)  # Numpy's seed
set_random_seed(random_state)  # Tensorflow's seed

def load_dataset():
    ''' Dataset Loading '''
    if dataset_name == "IMDb Large Movie Review Dataset":
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(seed=random_state, num_words=feature_count)

    elif dataset_name == "Movie Review Subjectivity Dataset":
        data = ["" for i in range(10000)]
        labels = ["" for i in range(10000)]
        count = 0
        with open('./gdrive/My Drive/Colab Datasets/Movie Review Subjectivity Dataset/plot.tok.gt9.5000', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line.rstrip('\n')
                labels[count] = 0
                count += 1
        with open('./gdrive/My Drive/Colab Datasets/Movie Review Subjectivity Dataset/quote.tok.gt9.5000', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line.rstrip('\n')
                labels[count] = 1
                count += 1   
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=random_state, shuffle=True)   
        del data, labels

    elif dataset_name == "Movie Review Polarity Dataset":
        data = ["" for i in range(10662)]
        labels = ["" for i in range(10662)]
        count = 0
        with open('./gdrive/My Drive/Colab Datasets/Movie Review Polarity Dataset/rt-polarity.neg', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line.rstrip('\n')
                labels[count] = 0
                count += 1
        with open('./gdrive/My Drive/Colab Datasets/Movie Review Polarity Dataset/rt-polarity.pos', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line.rstrip('\n')
                labels[count] = 1
                count += 1    
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=random_state, shuffle=True)  

    elif dataset_name == "SCH Dataset":
        df1 = pd.read_csv('./gdrive/My Drive/Colab Datasets/SCH Dataset/final_data_q13.csv', delimiter="|", header=None)
        df2 = pd.read_csv('./gdrive/My Drive/Colab Datasets/SCH Dataset/final_data_q14.csv', delimiter="|", header=None)
        df3 = pd.read_csv('./gdrive/My Drive/Colab Datasets/SCH Dataset/final_data_q15.csv', delimiter="|", header=None)
        df4 = pd.read_csv('./gdrive/My Drive/Colab Datasets/SCH Dataset/final_data_q16.csv', delimiter="|", header=None)

        df =  df1.append([df2, df3, df4], ignore_index=True)
        df.rename(columns={0: "Labels", 1: "Data_GR", 2: "Data_ENG"}, inplace=True)
        df.dropna(0, inplace=True)    
        df["Labels"] = df.apply(lambda row: row['Labels'] if isinstance(row['Labels'], str)==True else str(int(row['Labels'])), axis=1)
        df = df.sample(frac=1., random_state=random_state).reset_index(drop=True)

        print(f"--Dataset Info:\n{df.describe(include='all')}\n\n{df.head(4)}\n\n{df.iloc[:,0].value_counts()}\n--\n")                  
        
        train_data, test_data, train_labels, test_labels = train_test_split(df.loc[:,"Data_ENG"].values, df.loc[:,"Labels"].values, test_size=0.1, random_state=random_state, shuffle=True)

    elif dataset_name == "Finegrained Sentiment Dataset":
        data = [[] for i in range(294)]
        labels = ["" for i in range(294)]
        count = 0
        with open('./gdrive/My Drive/Colab Datasets/Finegrained Sentiment Dataset/finegrained.txt', 'r', encoding='iso-8859-15') as file:
            for line in file:
                if len(line.split("_")) == 3:
                    labels[count] = line.split("_")[1]                  
                elif len(line.strip()) == 0:
                    data[count] = ' '.join(data[count])
                    count += 1
                else:
                    temp = [x.strip() for x in line.split("\t")]
                    if len(temp[1]) > 1:
                        # "nr" label is ignored
                        if temp[0] in ["neg", "neu", "pos", "mix"]:
                            data[count].append(temp[0])              
        
        encoder = LabelEncoder()
        encoder.fit(labels)
        labels = encoder.transform(labels)
            
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=random_state, shuffle=True)   
        del data, labels 

    else:
        raise ValueError(f"Dataset is not implemented yet")

    # Print Dataset Information
    print(f"{dataset_name} Loaded. Training entries: {len(train_data)}, labels: {len(train_labels)}")
    for i in range(4): print(train_data[i], train_labels[i])
    print()

    return train_data, test_data, train_labels, test_labels

# NLP Functions
def imdb_specific_word_index():
    ''' A dictionary mapping words to an integer index '''
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()}
    #word_index["<PAD>"] = 0  # not inserting this key makes the word_index have the same length as in non-IMDb datasets, i.e. for 20000 features we end up with a length of 19999
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown/OOV word
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    return word_index, reverse_word_index
    
def other_datasets_word_index(tokenizer):
    ''' A dictionary mapping words to an integer index '''
    word_index = tokenizer.word_index
    
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    return word_index, reverse_word_index

def reduce_word_index_features(word_index, reverse_word_index, train_data):
    ''' Update the word index from its generic version to one that matches the number of features we selected '''
    word_index = {k: v for k, v in word_index.items() if v < feature_count} 
    reverse_word_index = {k: v for k, v in reverse_word_index.items() if k < feature_count} 

    print("\n".join([decode_review(instance, reverse_word_index, mode="join") for instance in train_data[0:4]]))

    return word_index, reverse_word_index

def sequence_padding(train_data, test_data, embeddings_sequence_length):
    train_data = pad_sequences(train_data,
                               padding='pre',  # Using 'pre' instead of 'post' on truncating leads to higher accuracy
                               truncating='pre',
                               maxlen=embeddings_sequence_length)

    test_data = pad_sequences(test_data,
                              padding='pre',
                              truncating='pre',
                              maxlen=embeddings_sequence_length)
    
    return train_data, test_data   

def print_dataset_length_stats(train_data):        
    maxim = 0
    total = 0
    count = 0
    for i in train_data:
        length = len(i)
        if length > maxim:
            maxim = length
        count += 1
        total += length
    print(f"General stats regarding the length of instances of the dataset (to help choose embeddings_sequence_length) - avg:{total/count:.2f} max:{maxim}\n")   

def decode_review(text, reverse_word_index, mode):
    if mode == "join":
        return ' '.join([reverse_word_index.get(i, '?') for i in text])
    else:
        return [reverse_word_index.get(i, '?') for i in text]

def encode_review(text, word_index):
    text = text.split()
    return [word_index.get(i, 2) for i in text]  # "2" refers to unknown/OOV word

"""**Regarding Word2Vec, Embeddings and Google Drive**

Converting the Word2Vec bin file to a text file of bigger size is even worse since we are using Google Drive.  
Ideally the Google Drive file must be as small as possible because the bottleneck is caused by Download Speed not by the loading process itself.

Location: USA  
Download: 134.43 Mbit/s  
Upload: 178.00 Mbit/s

Load time for the zipped file is 170 to 293 seconds.

---

So here is a great idea: download the file from Drive to Colab [[1](https://stackoverflow.com/questions/48735600/file-download-from-google-drive-to-colaboratory)] [[2](https://medium.freecodecamp.org/how-to-transfer-large-files-to-google-colab-and-remote-jupyter-notebooks-26ca252892fa)] at the very start so that we can maintain access for the entire Runtime.

Download time is 73 to 80 seconds.

---

**Regarding alternatives of Word2Vec**

FastText is simply Word2Vec with subword n-grams but requires more RAM and training time

---

**Regarding simple Word2Vec similarity examples (line 25)**

Other than the 3.6GB of RAM that we are using to load the Word2Vec embeddings, we can use simple similarity examples on the model to ensure its working will lead to an extra 3.6GB being used [[1](https://stackoverflow.com/questions/50478046/memory-error-when-using-gensim-for-loading-word2vec)] [[2](https://github.com/RaRe-Technologies/gensim/issues/293#issuecomment-175026483)].  
"I can load the model fine, and can retrieve the word vectors of words fine, but it seems to explode just when I try to access similarity-related functions (including 'doesn't match') etc."  

If you want to enable the similarity examples, change line 25.
"""

# Word Embeddings Functions
def load_word2vec_pretrained():  
    ''' Loads word2vec files from Google Drive, which are used later on '''
    time_counter = time.time()
    print(f"Downloading data from Google Drive to Colab hard drive...")
    #!mkdir my_data
    #!cp -i '/content/gdrive/My Drive/Colab Datasets/GoogleNews-vectors-negative300.bin.gz' '/content/my_data/GoogleNews-vectors-negative300.bin.gz'
    print(f"Download completed in {time.time()-time_counter:.2f}sec, displaying information")
    #!ls '/content/my_data' -l --block-size=MB

    time_counter = time.time()
    print(f"Loading file...")
    word2vec = KeyedVectors.load_word2vec_format('/content/my_data/GoogleNews-vectors-negative300.bin.gz', binary=True, unicode_errors='strict')    
    print(f"Loading completed in {time.time()-time_counter:.2f}sec")
    
    # Download, Unzip, Convert from bin to text file, Upload
    # word2vec = gzip.open('/content/gdrive/My Drive/Colab Datasets/GoogleNews-vectors-negative300.bin.gz', 'rb')
    # word2vec.save_word2vec_format('/content/gdrive/My Drive/Colab Datasets/GoogleNews-vectors-negative300.txt', binary=False)
    # word2vec_text_mode = KeyedVectors.load_word2vec_format('/content/gdrive/My Drive/Colab Datasets/GoogleNews-vectors-negative300.txt')    
    
    dog = word2vec['dog']
    print(f"\nWord2Vec Embeddings Dimension: {dog.shape}")
    print(f"Example Values: {dog[:10]}")
    
    if False:
        # Some predefined functions that show content related information for given words
        print(f"Test 1: {word2vec.most_similar(positive=['woman', 'king'], negative=['man'])}")
        print(f"Test 2: {word2vec.most_similar('hyundai', topn=10)}")
        print(f"Test 3: {word2vec.doesnt_match('breakfast cereal dinner lunch'.split())}")  # Raises Warning
        print(f"Test 4: {word2vec.similarity('woman', 'man')}")
    else:
        print(f"Examples of Word2Vec and similarity function usage won't be run in order to preserve RAM.\n")
        
    return word2vec

def find_common_words(embeddings_vocab, original_vocab):
    ''' Find words in common between two vocabularies '''
    return set(filter(lambda x: x in embeddings_vocab, original_vocab))  # filter is not a good format, if the variable is used once it becomes empty (i.e. a generator), use set instead

def find_oov_words(embeddings_vocab, original_vocab):
    ''' Find words not in common between two vocabularies (OOV words) '''
    return set(filter(lambda x: x not in embeddings_vocab, original_vocab))  # filter is not a good format, if the variable is used once it becomes empty (i.e. a generator), use set instead

def manage_oov_words(embeddings, vocab):
    ''' Find words in common, meaning the remaining words are out-of-vocabulary (OOV) ones '''
    words_oov = find_oov_words(embeddings.vocab, vocab)  # Find words in common, meaning the remaining words are out-of-vocabulary (OOV) ones
    print(f"Impossible to remove all occurances of {len(words_oov)} words since time complexity would be too high. Out-of-vocabulary words will be kept and will be assigned vectors...")
    if len(words_oov) >= 3:
        to_print = iter(words_oov)
        print(f"Some examples: {next(to_print)}, {next(to_print)}, {next(to_print)}")
    
    #encoding_oov = [vocab[word] for word in words_oov]
    #for instance in data_container:
        #[word for word in instance if word not in encoding_oov]  # looking up a set is the absolute fastest in python

def assign_embeddings(embeddings, embeddings_dimension, vocab, mode):
    ''' Create an embeddings (weight) matrix for the Embedding layer from a loaded embedding ''' 
    words_in_common = find_common_words(embeddings.vocab, vocab)
    words_oov = find_oov_words(embeddings.vocab, vocab)  # Find words in common, meaning the remaining words are out-of-vocabulary (OOV) ones
    print(f"Number of vocabulary words that cannot be found in the Word2Vec embeddings: {len(words_oov)}")  
    
    # "Total vocabulary size plus 0 for unknown words 'len(vocab) + 1" is not entirely true, the index of 0 is simply not used leading to, for example a length of 19999 for 20000 features:
    vocab_size = len(vocab) + 1
    # Initialize the weight matrix with 0s
    embed_final_matrix = np.zeros((vocab_size, embeddings_dimension))
    # Store vectors using an integer mapping, for example from the Tokenizer
    
    if mode == "zeros":
        for word in words_in_common:
            embed_final_matrix[vocab[word]] = embeddings[word]
    elif mode == "random":
        np.random.seed(random_state)
        for word, i in vocab.items():
            if word in words_oov:
                embed_final_matrix[i] = np.random.uniform(low=-0.5, high=0.5, size=embeddings_dimension)
            else:
                embed_final_matrix[i] = embeddings[word]
    else:
        raise ValueError(f"{mode} is not a valid mode parameter.")
    
    print(f"Embeddings assignment completed.\n")      
    return embed_final_matrix

## Evaluate on External Data

### Preprocessing Run

### FIRST, execute the entire preprocessing phase again (there are more efficient ways, but this works universally).


def run_preprocessing(input_test_sentence):
    ###   INPUT    ###
    #test_sentence = ["very bad movie, wow awful such a bad performance from the actors."]  # Example sentence for debug purposes
    test_sentence = input_test_sentence
    ###            ###

    ### Parameters ###
    remove_first = False #@param {type:"boolean"}
    embeddings_mode = "Word2Vec Pretrained" #@param ["One-hot Encoding", "Tokenizing", "Word2Vec Pretrained", "Word2Vec Training"]
    embeddings_sequence_length = 50 #@param {type:"integer"}
    trainable = False #@param {type:"boolean"}
    outofvocab_mode = "random" #@param ["zeros", "random"]
    ###            ###

    # Load Dataset
    train_data, test_data, train_labels, test_labels = load_dataset()
    print_dataset_length_stats(train_data)

    # Remove the <START> symbol from all instances
    if remove_first == True:
        if train_data[0][0] == word_index["<START>"]:
            for i in range(len(train_data)):
                train_data[i] = train_data[i][1:]
            for i in range(len(test_data)):
                test_data[i] = test_data[i][1:]

    # Create the Embeddings            
    if embeddings_mode == "One-hot Encoding":  
        ''' 
            ONE-HOT ENCODING
            Description: Not traditional One-hot, but instead [3, 62, 5, 90, ...] 
            Embedding_Layer: Yes
        '''   
        if dataset_name != "IMDb Large Movie Review Dataset":  # keras.imdb is already One-hot Encoded
            tokenizer = Tokenizer(num_words=feature_count, 
                                lower=True, 
                                split=' ', 
                                oov_token="<UNK>")
            tokenizer.fit_on_texts(train_data)
            # 'texts_to_sequences' list of strings as input and sequence of integers as output, 'texts_to_matrix' is meant to return a matrix of counts/tf-idfs
            train_data = tokenizer.texts_to_sequences(train_data)
            test_data = tokenizer.texts_to_sequences(test_data)  
        
        # Word_Index Stuff
        if dataset_name == "IMDb Large Movie Review Dataset":  # keras.imdb is already One-hot Encoded
            word_index, reverse_word_index = imdb_specific_word_index()
        else:
            word_index, reverse_word_index = other_datasets_word_index(tokenizer)

        word_index, reverse_word_index = reduce_word_index_features(word_index, reverse_word_index, train_data)  # Update the word index to match the number of features we selected       

        # NEW
        test_sentence = encode_review(test_sentence[0], word_index)
        test_sentence = np.array([test_sentence])  # Convert back to 2D array even if there is no use yet

        # Peform Sequence Padding
        train_data, test_data = sequence_padding(train_data, test_data, embeddings_sequence_length)
        # NEW
        test_sentence, _ = sequence_padding(test_sentence, test_sentence, embeddings_sequence_length) 

    elif embeddings_mode == "Tokenizing":  
        ''' 
            TOKENIZING
            Description: Unlike other modes this isn't exactly an embedding, leads to a collection of floats [0.00, 0.02, 0.12, 0.04, ...] based on tf-idf 
            Embedding_Layer: No
        '''    
        if dataset_name == "IMDb Large Movie Review Dataset":  # keras.imdb is already One-hot Encoded so let's transform it back to text
            word_index, reverse_word_index = imdb_specific_word_index()
            train_data = [decode_review(instance, reverse_word_index, mode="join") for instance in train_data]
            test_data = [decode_review(instance, reverse_word_index, mode="join") for instance in test_data]
        
        tokenizer = Tokenizer(num_words=feature_count, 
                            lower=True, 
                            split=' ', 
                            )
        tokenizer.fit_on_texts(train_data)
        # 'texts_to_matrix' list of strings as input, 'sequences_to_matrix' list of integer word indices as input 
        train_data = tokenizer.texts_to_matrix(train_data, mode='tfidf')
        test_data = tokenizer.texts_to_matrix(test_data, mode='tfidf')

        # NEW
        test_sentence = tokenizer.texts_to_matrix(test_sentence, mode='tfidf')
    
    elif embeddings_mode == "Word2Vec Pretrained":
        ''' 
            WORD2VEC PRETRAINED
            Description: A much more advanced form of embeddings that is created through training a model unlike previous modes. Implements the CBOW and the Skip-gram models in order to learn word embeddings.
            Embedding_Layer: Yes
        '''
        # RUNNING word2vec is not actually needed, only needed for training the neural network
        
        if dataset_name != "IMDb Large Movie Review Dataset":  # keras.imdb is already One-hot Encoded
            tokenizer = Tokenizer(num_words=feature_count, 
                                lower=True, 
                                split=' ', 
                                oov_token="<UNK>")
            tokenizer.fit_on_texts(train_data)
            # 'texts_to_sequences' list of strings as input and sequence of integers as output, 'texts_to_matrix' is meant to return a matrix of counts/tf-idfs
            train_data = tokenizer.texts_to_sequences(train_data)
            test_data = tokenizer.texts_to_sequences(test_data)   
        
        # Word_Index Stuff
        if dataset_name == "IMDb Large Movie Review Dataset":  # keras.imdb is already One-hot Encoded
            word_index, reverse_word_index = imdb_specific_word_index()
        else:
            word_index, reverse_word_index = other_datasets_word_index(tokenizer)

        word_index, reverse_word_index = reduce_word_index_features(word_index, reverse_word_index, train_data)  # Update the word index to match the number of features we selected       

        # NEW
        test_sentence = encode_review(test_sentence[0], word_index)
        test_sentence = np.array([test_sentence])  # Convert back to 2D array even if there is no use yet

        # Peform Sequence Padding  
        train_data, test_data = sequence_padding(train_data, test_data, embeddings_sequence_length)
        # NEW
        test_sentence, _ = sequence_padding(test_sentence, test_sentence, embeddings_sequence_length)  

    elif embeddings_mode == "Word2Vec Training":
        ''' 
            Description: A much more advanced form of embeddings that is created through training a model unlike previous modes. Implements the CBOW and the Skip-gram models in order to learn word embeddings.
            Embedding_Layer: Yes
        '''
        if dataset_name != "IMDb Large Movie Review Dataset":  # keras.imdb is already One-hot Encoded so let's transform it back to text
            tokenizer = Tokenizer(num_words=feature_count, 
                                lower=True, 
                                split=' ', 
                                oov_token="<UNK>")
            tokenizer.fit_on_texts(train_data)
            # 'texts_to_sequences' list of strings as input and sequence of integers as output, 'texts_to_matrix' is meant to return a matrix of counts/tf-idfs
            train_data = tokenizer.texts_to_sequences(train_data)
            test_data = tokenizer.texts_to_sequences(test_data)  
            
        # Word_Index Stuff
        if dataset_name == "IMDb Large Movie Review Dataset":  # keras.imdb is already One-hot Encoded so let's transform it back to text
            word_index, reverse_word_index = imdb_specific_word_index()
        else:
            word_index, reverse_word_index = other_datasets_word_index(tokenizer)

        word_index, reverse_word_index = reduce_word_index_features(word_index, reverse_word_index, train_data)  # Update the word index to match the number of features we selected        

        # NEW
        test_sentence = encode_review(test_sentence[0], word_index)
        test_sentence = np.array([test_sentence])  # Convert back to 2D array even if there is no use yet

        # Peform Sequence Padding  
        train_data, test_data = sequence_padding(train_data, test_data, embeddings_sequence_length)
        # NEW
        test_sentence, _ = sequence_padding(test_sentence, test_sentence, embeddings_sequence_length)          

    # Print the resulting instances       
    for i in range(4): print(type(train_data[i]), list(train_data[i]))

    return test_sentence

"""Decide on a final embeddings mode out of the 4. Then the above code can be run once at the start, and a different code can be run each time for a prediction.

### Code
"""

# Evaluation Phase
def evaluate_single_sentence(model, test_sentence, multiclass):
    if multiclass == False: 
        probability = model.predict(test_sentence)
        predictions = [1 * (x[0]>=0.5) for x in probability]    
    else:    
        probability = model.predict(test_sentence)
        predictions = np.argmax(probability, axis=1)
    
    # 0 stands for negative, 1 stands for positive
    if predictions[0] == 1:
        return ("positive", probability) 
    elif predictions[0] == 0:
        return ("negative", probability)
    else:
        return ("no prediction", None)

#print(evaluate_single_sentence(model, test_sentence, multiclass))