"""
This file contains all the methods that preprocesses the data, loads the model,
then runs the model on the data inputs given by the user.

@author: YuanHu
"""
import re
import unicodedata
import random
from pickle import load
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import tensorflow as tf
# print(tf.__version__)

# Load the model
with open("model/model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
# load weights into the model
loaded_model.load_weights("model/model_weights.h5")
# load tokenizer
tokenizer = load(open('model/tokenizer.pkl', 'rb'))

# Text Cleaning
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        if word != ',' or word != '.':
          new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
          new_words.append(new_word)
    return new_words

def clean_text(text):
    # noise removal
    text = re.sub(r'\bhttps?://\w+.+[^ ]\b', 'link', text) #replace url with "link"
    text = re.sub(r'[a-zA-Z0-9]*\.?github\.?[a-zA-Z0-9]*','github',text) #replace github links with "github"
    text = re.sub(r'\~\\cite\{[^}]*\}','cite',text) # remove cite in the format of "~\cite{DeSaOR16}"
    text = re.sub(r'\[[^]]*\]', '', text) # remove between square brackets
    text = re.sub(r'\([^)]*\)', '', text) # remove between parentheses
    text = re.sub(r'\{[^)]*\}', '', text) # remove between curly brackets

    # normalization
    text = text.lower() # convert to lowercase text

    text = re.sub(r'\-',' ', text) # seperate words like 'video-related'
    text = re.sub(r'\/',' ', text) # seperate words like 'descriptors/tags'
    text = re.sub(r'[^a-zA-Z0-9\s\.\,]', '', text) # remove punctuation
    text = re.sub(r'seq2seq', 'seqtoseq', text)

    text = re.sub(r'[-+]?\d*\.?\d+', 'NUMBER', text) # replace numbers with "NUMBER"

    text = re.sub(r'\.{2,}', '', text) # remove '..','...'
    text = re.sub(r'\.', ' . ', text) # seperate '.' from text
    text = re.sub(r'\,' , ' , ', text) # seperate ',' from text

    text = ' '.join(remove_non_ascii(text.split())) # remove non-ascii words
    text = re.sub(r'[\w]*NUMBER[\w]*', 'NUMBER', text) # replace anyword containing "NUMBER" with "NUMBER"

    # convert from British English to American English
    text = re.sub(r'modelled', 'modeled', text)
    text = re.sub(r'modelling', 'modeling', text)
    text = re.sub(r'parallelisation', 'parallelization', text)
    text = re.sub(r'parallelising', 'parallelizing', text)
    text = re.sub(r'analysed', 'analyzed', text)
    text = re.sub(r'generalised', 'generalized', text)
    text = re.sub(r'maximisation', 'maximization', text)
    text = re.sub(r'recogniser', 'recognizer', text)
    text = re.sub(r'optimised', 'optimized', text)
    text = re.sub(r'analyse', 'analyze', text)
    text = re.sub(r'generalisation', 'generalization', text)
    text = re.sub(r'generalise', 'generalize', text)
    text = re.sub(r'factorisation', 'factorization', text)
    text = re.sub(r'behaviour', 'behavior', text)
    text = re.sub(r'interpretted', 'interpreted', text)
    text = re.sub(r'neighbouring', 'neighboring', text)
    text = re.sub(r'neighbour', 'neighbor', text)
    text = re.sub(r'neighbours', 'neighbors', text)
    text = re.sub(r'dependant', 'dependent', text)
    text = re.sub(r'localisation', 'localization', text)
    text = re.sub(r'amortised', 'amortized', text)
    text = re.sub(r'amortisation', 'amortization', text)
    text = re.sub(r'neutralising', 'neutralizing', text)
    text = re.sub(r'prioritised', 'prioritized', text)
    text = re.sub(r'characterised', 'characterized', text)
    text = re.sub(r'characterise', 'characterize', text)
    text = re.sub(r'centeralised', 'centeralized',text)
    text = re.sub(r'initialisation', 'initialization', text)
    text = re.sub(r'initialised', 'initialized', text)
    text = re.sub(r'regularisation', 'regularization', text)
    text = re.sub(r'regularised', 'regularized', text)
    text = re.sub(r'optimisation', 'optimization', text)
    text = re.sub(r'optimise', 'optimize', text)
    text = re.sub(r'minimisation', 'minimization', text)
    text = re.sub(r'generalises', 'generalizes', text)
    text = re.sub(r'parameterised', 'parameterized', text)
    text = re.sub(r'parameterises', 'parameterizes', text)
    text = re.sub(r'reparameterisation', 'reparameterization', text)
    text = re.sub(r'optimising', 'optimizing', text)
    text = re.sub(r'favourable', 'favorable', text)
    text = re.sub(r'hypothesised', 'hypothesized', text)
    text = re.sub(r'summarise', 'summarize', text)
    text = re.sub(r'standardised', 'standardized', text)
    text = re.sub(r'randomisation', 'randomization', text)
    text = re.sub(r'synchronisation', 'synchronization', text)
    text = re.sub(r'travelling', 'traveling', text)
    text = re.sub(r'maximising', 'maximizing', text)
    text = re.sub(r'initialising', 'initializing', text)
    text = re.sub(r'initialise', 'initialize', text)
    text = re.sub(r'localise', 'localize', text)
    text = re.sub(r'localising', 'localizing', text)
    text = re.sub(r'localisation', 'localization', text)
    text = re.sub(r'parametrised', 'parametrized', text)
    text = re.sub(r'factorised', 'factorized', text)
    text = re.sub(r'factorisation', 'factorization', text)

    return text

def text_recover(text):
    text = re.sub(r' \.', '.' , text)
    text = re.sub(r' \,' , ',' , text)
    number = str(random.randint(3,20))
    text = re.sub(r'NUMBER', number, text)

    return text
# Generate Text
# generate a sequence from a language model
def sequence_generator(model, tokenizer, seq_length, seed_text, n_words):
    """
    Generate text with length of n_words, given the model, tokenizer, seed and seq_length
    """
    seed_text = seed_text.lower()
    result = []
    word_index = tokenizer.word_index
    text_ip = seed_text

    for _ in range(n_words):
      # tokenization
      sequences = tokenizer.texts_to_sequences([text_ip])[0]
      # truncate sequences to a fixed length
      sequences = pad_sequences([sequences], maxlen=seq_length, truncating='pre')
      # predict probabilities for each word
      y_predict = model.predict_classes(sequences, verbose=0)
      # map word index to word
      word_op = ''
      for word, index in word_index.items():
        if index == y_predict:
          word_op = word
          break
		# update text input
      text_ip += ' ' + word_op

      result.append(word_op)
    return ' '.join(result)

def text_generator(text):
    """
    Calls on previous functions to generate texts.
    Takes a string as the seed, and generate text with length of n_words
    """
    seed = clean_text(text)

    generated_text = sequence_generator(loaded_model, tokenizer, 99, seed, 50)

    generated_text = text_recover(generated_text)
    seed = text_recover(seed)

    return generated_text

if __name__ == '__main__':
    text = input("Input text: ")

    print("Generated text: ")
    print(text_generator(text))
