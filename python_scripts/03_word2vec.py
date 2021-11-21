import pandas as pd
import os
import collections
import csv
import logging
import numpy as np
import datetime as datetime
import types

from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, Concatenate, dot
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins import projector

df = pd.read_pickle('./data/df_processed_bigrams.pickle')

# Note to do - need to add time element

def log_newline(self, how_many_lines=1):
    file_handler = None
    if self.handlers:
        file_handler = self.handlers[0]

    # Switch formatter, output a blank line
    file_handler.setFormatter(self.blank_formatter)
    for i in range(how_many_lines):
        self.info('')

    # Switch back
    file_handler.setFormatter(self.default_formatter)

def logger_w2v():
    
    log_file = os.path.join('./data', 'word2vec.log')
    print('log file location: ', log_file)
    
    log_format= '%(asctime)s - %(levelname)s - [%(module)s]\t%(message)s'
    formatter = logging.Formatter(fmt=(log_format))
    
    fhandler = logging.FileHandler(log_file)
    fhandler.setFormatter(formatter)
    
    logger = logging.getLogger('word2vec')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fhandler)
    logger.default_formatter = formatter
    logger.blank_formatter = logging.Formatter(fmt="")
    logger.newline = types.MethodType(log_newline, logger)
    
    return logger

class Word2Vec:
    """
    apply word2vec to text
    """

    def __init__(self, logger, vocab_size, vector_dim, input_target, input_context,
                 load_pretrained_weights, weights_file_name, train_model_flag, checkpoint_file):
        """
        Args:
            vocab size: integer of number of words to form vocabulary from
            vector_dim: integer of number of dimensions per word
            input_target: tensor representing target word
            input_context: tensor representing context word
        """
        self.logger = logger        
        self.vocab_size = vocab_size
        self.vector_dim = vector_dim
        self.input_target = input_target
        self.input_context = input_context
        self.load_pretrained_weights = load_pretrained_weights
        self.weights_file_name = weights_file_name
        self.checkpoint_file = checkpoint_file
        self.train_model_flag = train_model_flag
        self.model = self.create_model()
        
    def build_dataset(self, words):
        """
        :process raw inputs into a dataset

        Args:
            words: list of strings

        Returns:
            tuple:
                data: list of integers representing words in words
                count: list of count of most frequent words with size n_words
                dictionary: dictionary of word to unique integer
                reverse dictionary: dictionary of unique integer to word
        """
        self.logger.info("Building dataset")

        count = [['UNK', -1]]
        words = [item for sublist in words for item in sublist]
        print(len(words))
        count.extend(collections.Counter(words).most_common(self.vocab_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0        
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.dictionary = dictionary

        # Save dictionary
        dict_path = './data'
        dict_file = 'dictionary.csv'
        dict_file = os.path.join(dict_path,dict_file)
        
        with open(dict_file, 'w') as f:
            for key in dictionary.keys():
                f.write("%s,%s\n"%(key,dictionary[key]))

        return data, count, dictionary, reversed_dictionary
    
    def get_training_data(self, data, window_size):
        """
        :create text and label pairs for model training

        Args:
            data: list of integers representing words in words
            window_size: integer of number of words around the target word that
                         will be used to draw the context words from.

        Returns:
            tuple:
                word_target: list of arrays representing target word 
                word_context: list of arrays representing context word in 
                              relation to target word
                labels: list containing 1 for true context, 0 for false context
                couples: list of pairs of word indexes aligned with labels
        """
        # the probability of sampling the word i-th most common word 
        sampling_table = sequence.make_sampling_table(self.vocab_size)
        
        self.logger.info("finding training data with labels")
        couples, labels = skipgrams(data, self.vocab_size, window_size=window_size, 
                                    sampling_table=sampling_table)

        print('couples length', len(couples))
        print(couples[0])
        print(couples[1000])
        self.logger.info("define target and context variables")
        #word_target, word_context = zip(*couples)
        word_target = [c[0] for c in couples]
        word_contextt = [c[1] for c in couples]
        self.logger.info("converting to numpy arrays")
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        
        self.logger.info("training data acquired")

        return word_target, word_context, labels

    def create_model(self):
        """
        :keras functional API and embedding layers

        Returns:
            model: untrained word2vec model
        """

        # embedding layer
        embedding = Embedding(self.vocab_size, self.vector_dim, input_length=1, name='embedding')

        # embedding vectors
        target = embedding(self.input_target)
        target = Reshape((self.vector_dim, 1))(target)
        context = embedding(self.input_context)
        context = Reshape((self.vector_dim, 1))(context)

        # dot product operation to get a similarity measure
        dot_product = dot([target, context], axes=1, normalize=False)
        dot_product = Reshape((1,))(dot_product)

        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)

        # create the training model
        self.model = Model(inputs=[self.input_target, self.input_context], outputs=output)

        return self.model

    def train_model(self, epochs, batch_size, word_target, word_context, labels):
        """
        :trains word2vec model

        Args:
            model: word2vec model
            epochs: integer of number of iterations to train model on
            batch_size: integer of number of words to pass to epoch
            word_target: list of arrays representing target word 
            word_context: list of arrays representing context word in relation 
                          to target word
            labels: list containing 1 for true context, 0 for false context

        Returns:
            model: trained word2vec model
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        #loss = tf.keras.losses.BinaryCrossentropy()
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)

        # tensorboard callback
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir='tensorboard_log/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)

        if self.load_pretrained_weights:
            self.load_prior_weights()
            if not self.train_model_flag:
                return self.model

        arr_1 = np.zeros((batch_size,))
        arr_2 = np.zeros((batch_size,))
        arr_3 = np.zeros((batch_size,))
        for i in range(epochs):
            idx = np.random.choice(list(range(len(labels))), size=batch_size, replace=False)
            arr_1[:] = np.array([word_target[i] for i in idx])
            arr_2[:] = np.array([word_context[i] for i in idx])
            arr_3[:] = np.array([labels[i] for i in idx])
            loss = self.model.train_on_batch([arr_1, arr_2], arr_3)
            with summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=i)
            if (i+1) % 500 == 0:
                print("Iteration {}, loss={}".format(i+1, loss))
            if (i+1) % 1000 == 0:
                checkpoint_dir = './model/model_weights'
                checkpoint_file = f"cp-epoch-{i+1:010d}.h5"
                checkpoint_path = os.path.join(checkpoint_dir,checkpoint_file)
                self.model.save_weights(checkpoint_path)
                self.embedding_projector(log_dir)

        return self.model
    
    def embedding_projector(self, log_dir):
        """
        :visualise embeddings in tensorboard
        """
        # Save Labels separately on a line-by-line manner.
        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            for subwords in self.dictionary.keys():
                f.write("{}\n".format(subwords))
            # Fill in the rest of the labels with "unknown"
            for unknown in range(1, self.vocab_size - len(self.dictionary.keys())):
                f.write("unknown #{}\n".format(unknown))

        # Save the weights we want to analyse as a variable. 
        weights = tf.Variable(self.model.layers[2].get_weights()[0])
        checkpoint_w = tf.train.Checkpoint(embedding=weights)
        checkpoint_w.save(os.path.join(log_dir, "embedding.ckpt"))

        # Set up config
        config_tb = projector.ProjectorConfig()
        embedding_tb = config_tb.embeddings.add()
        embedding_tb.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding_tb.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(log_dir, config_tb)
        
        
    def load_prior_weights(self):
        """
        :load prior weights if load_pretrained_weights = True in main file
        """ 
        #abs_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
        #checkpoint_dir = os.path.join(abs_path, self.config['model']['model_dir'], self.config['model']['model_weights'])
        #checkpoint_path = os.path.join(checkpoint_dir,self.checkpoint_file)
        checkpoint_dir = './model/model_weights'
        checkpoint_file = self.weights_file_name
        checkpoint_path = os.path.join(checkpoint_dir,checkpoint_file)
        self.model.load_weights(checkpoint_path)
        self.logger.info('Loaded pre trained wweights from {}'.format(str(checkpoint_path)))

def get_word_vectors(model):

    embedding_weights = model.layers[2].get_weights()[0]
    #word_embeddings = {w:embedding_weights[idx] for w, idx in dictionary.items()}
    
    return embedding_weights

def tokenise_dataset(df):

    tokens = df['content_processed'].str.split(" ")

    return tokens

words = tokenise_dataset(df)

logger = logger_w2v()

vocab_size = 10000
vector_dim = 300
input_target = Input((1,))
input_context = Input((1,))
load_pretrained_weights = False
weights_file_name = f"cp-epoch-0000010000-B1000.h5"
checkpoint_file = None
train_model_flag = True

word2vec = Word2Vec(logger, vocab_size, vector_dim, input_target, input_context,
                    load_pretrained_weights, weights_file_name, train_model_flag, checkpoint_file)

data, count, dictionary, reversed_dictionary = word2vec.build_dataset(words)

print(f'dictionary length is: {len(dictionary)}')
print(f'data length is: {len(data)}')

window_size = 3

word_target, word_context, labels = word2vec.get_training_data(data, window_size)

np.save('word_target.csv', word_target)
