# Text Generation

## Objective
The goal of this project is to generate scientific sentences, when a scientific phrase or sentence is given. 

## Background
Text generation is an application of language modeling, and a subfield of natural language processing. It utilizes techniques in artificial intelligence to automatically generate natural language text, which fits in a certain communication context.

Text generation can be used to write stories, poems, emails, news articles, and more. It is also useful for machine translation and chatbots.


## Descriptions
Note: Due to the lack of computing power, only the abstracts of the articles are selected to form the corpus.

#### `scraper.py` ####
This file is used to collect papers information from NIPS website using scraping techniques.

#### `text_preprocessing.jpynb` ####
It is used for data cleaning, and save clean text into .txt file for model training.

Steps involved in text preprocessing:
* Words in British English are converted into American English.

**Note**: Although included in the data cleaning pipeline to reduce the vocabulary size , the following steps can be skipped for large corpus and with enough computing power.

* URLs, equations, citations are removed.
* Punctuation (except periods and commas), special characters are removed. 
* Hyphenated descriptions like “video-related” are converted into separate words, like “video related”.
* All numbers are replaced with “NUMBER”. 
* All characters are converted to lowercase. 

#### `text_generation_word_train.ipynb` ####
It is used for data preparation, and data modeling. It outputs a language model for text generation.

Bidirectional LSTM is used to improve the performance.

A Python generator is used to save memory space, and solve scaling problem.

#### `text_generation_word_generate.ipynb` ####
It loads a pre-trained model created from `text_generation_word_train.ipynb` to generate text. A seed of a text sequence needs to be provided so that the model can generate text from there.

#### `gpt2_train.ipynb` and `gpt2_generate.jpynb` ####
GPT-2 model is used to evaluate the performance of the developed LSTM model.

`gpt2_train.jpynb` takes the cleaned text file from `text_preprocessing.jpynb`,  and uses GPT-2 to retrain a model. Then it passes the trained model to `gpt2_generate.jpynb` to generate text.

For more details regarding how to use a GPT-2 model, please refer to [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)

## Future work
Several ways that can further improve the model:
* In text preprocessing part, add spell checker
* Use the entire articles to form the corpus
* Do rigorous parameters tuning 
* Add more LSTM layers
* Try seq2seq model

## References:
[1] Andrej Karpathy, 2015, [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[2] Schuster, Mike, and Kuldip K. Paliwal, 1997,  [Bidirectional recurrent neural networks](https://www.researchgate.net/profile/Mike_Schuster/publication/3316656_Bidirectional_recurrent_neural_networks/links/56861d4008ae19758395f85c.pdf) 

[3]Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever,  2019, [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)






