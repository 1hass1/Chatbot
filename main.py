# Chatbot
# dataset from a movie conversation

# Importing Libraries
import numpy as np
import tensorflow as tf
import re
import time
from keras import layers
from keras.layers import Dense

# Part 1: Data Preprocessing


lines = open("movie_lines.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open("movie_conversations.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')


# creating a dictionary that maps each line with its ID
# in order to create a table with input and its output (target)
# to compare the target from the character with the target from the chatbot

id2line = {}  # initialize a dictionary
for line in lines:
  _line = line.split(' +++$+++ ')   # _ means its a temporary variable (local)
  if len(_line) == 5:
    id2line[_line[0]] = _line[4]
    
    
# make a list of all the conversations containing the lines id
conversations_ids = []
for conversation in conversations[:-1]:      # excluding last row
  _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
  conversations_ids.append(_conversation.split(','))

# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
  for i in range(len(conversation) - 1):
    questions.append(id2line[conversation[i]])
    answers.append(id2line[conversation[i+1]])

print("Num GPUs Available: ", len(tf.compat.v1.config.experimental.list_physical_devices('GPU')))

tf.compat.v1.debugging.set_log_device_placement(True)

try:
  # Specify an invalid GPU device
  with tf.compat.v1.device('/device:GPU:2'):
    a = tf.compat.v1.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.compat.v1.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
except RuntimeError as e:
  print(e)


tf.compat.v1.config.set_soft_device_placement(True)
tf.compat.v1.debugging.set_log_device_placement(True)

# Creates some tensors
a = tf.compat.v1.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.compat.v1.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.compat.v1.matmul(a, b)
print(c)



# Doing 1st cleaning of the texts
# lower case, appostrophies...
def clean_text(text):
  text = text.lower()
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"he's", "he is", text)
  text = re.sub(r"she's", "she is", text)
  text = re.sub(r"that's", "that is", text)
  text = re.sub(r"what's", "what is", text)
  text = re.sub(r"where's", "where is", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"can't", "cannot", text)
  text = re.sub(r"[-()\"#/&:;<>{}+=.?,]", "", text)
  return text


# apply this function to the questions and answers

# Cleaning questions
clean_questions = []
for question in questions:
  clean_questions.append(clean_text(question))


# Cleaning answers
clean_answers = []
for answer in answers:
  clean_answers.append(clean_text(answer))


# Now, we create a dictionary that maps each word to its
# number of  occurances
word2count = {}
for question in clean_questions:
  for word in question.split():
    if word not in word2count:
      word2count[word] = 1
    else:
      word2count[word] += 1
      
for answer in clean_answers:
  for word in answer.split():
    if word not in word2count:
      word2count[word] = 1
    else:
      word2count[word] += 1
  

# Now, tokenization and filtering the non frequent words
# its a very important step
# we create 2 dictionaries to map the question words
# and answer words to a unique integer
# plus, an if statement to check if a word occurances is 
# above or below a certain threshold

threshold = 20   # 20%

questionswords2int = {}
word_number = 0
for word, count in word2count.items():
  if count >= threshold:
    questionswords2int[word] = word_number
    word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
  if count >= threshold:
    answerswords2int[word] = word_number
    word_number += 1

# adding the last tokens to these 2 dictionaries
tokens = ['<PAD>','<EOS>', '<OUT>', '<SOS>']
for token in tokens:
  questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
  answerswords2int[token] = len(answerswords2int) + 1


# create the inverse dictionary for answerswords2int 
# to use in the seq2seq model
# its very important to do inverse mapping of a dictionary
# useful skill
# w_i is the value and w is the key in this new dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}


# now we add the EOS token to the end of every answer
# an EOS token is needed at the end of an answer
# for our seq-2-seq model
for i in range(len(clean_answers)):
  clean_answers[i] += " <EOS>"
  
  
# now we remove the 5% least frequent words from clean questions and clean answers lists
# then the final step: sorting the questions and answers by the length of the question
# (very important... to optimize the training performance)


# Translating all the questions and the answers into integers
# and replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
  ints = []
  for word in question.split():
    if word not in questionswords2int:
      ints.append(questionswords2int['<OUT>'])
    else:
      ints.append(questionswords2int[word])
  questions_into_int.append(ints)
  
answers_into_int = []
for answer in clean_questions:
  ints = []
  for word in answer.split():
    if word not in answerswords2int:
      ints.append(answerswords2int['<OUT>'])
    else:
      ints.append(answerswords2int[word])
  answers_into_int.append(ints)



# Sorting questions and answers by the length of questions
# this will speed up the training and reduce the loss
# since it will reduce the amount of padding during the training

sorted_clean_questions = []
sorted_clean_answers = []

# lets choose a short length of a question (25)
for length in range(1, 25 + 1):
# for each question with this length, we want to get 2 important elements
# the index of the question, and the question (translated into ints)
# to get them both at the same time, we use enumerate
# so we will enumerate the questions_into_int list
  for i in enumerate(questions_into_int):
    if len(i[1]) == length:
      sorted_clean_questions.append(questions_into_int[i[0]])
      sorted_clean_answers.append(answers_into_int[i[0]])







# Part 2: Building the SEQ2SEQ Model

# here, we use tensorflow to build the architecture of the model


# Creating placeholders for the inputs and the targets
# (tensorflow placeholders)
# to use them in future training

# create a placeholder for the input
def model_inputs():
  tf.compat.v1.disable_eager_execution()
  inputs = tf.compat.v1.placeholder(tf.int32, [None, None], name = 'input')
# 1st argument is type, 2nd is dimensions of the matrix of input (2D)
  targets = tf.compat.v1.placeholder(tf.int32, [None, None], name = 'target')
  
  # also we make a placeholder for the learning rate, and dropout rate
  lr = tf.compat.v1.placeholder(tf.float32, name = 'learning_rate')
  keep_prob = tf.compat.v1.placeholder(tf.int32, name = 'keep_prob')  # dropout
  return inputs, targets, lr, keep_prob



# Preprocessing the targets

# the structure:
# 1st, the targets must be into batches !
# the RNN of the decoder will not accept single targets (single answers)
# (targets are answers)
# ex: a batch of 10 rows of the answers will go into the decoder RNN
# plus, each answer must start with SOS token (should add it to the beginning to each answer)


def preprocess_targets(targets, word2int, batch_size):
  # this function will turn the answers into batches
  # and add the SOS token
  # word2int is the dictionary that maps the tokens to integers
  # (to get the unique identifier of the SOS token)
  
  # we want to remove the last column of the answers ids (dont need it)
  # and concatenate the 1st column of answers to be SOS (the unique identifier of SOS token)
  left_side = tf.fill([batch_size, 1], word2int['<SOS>'])  
  # batch_size is dimension, # columns is 1
  # fill it with the unique identifiers encoding the SOS
  
  right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
  preprocessed_targets = tf.concat([left_side, right_side], 1)
  return preprocessed_targets
  


# now, we create the encoding and decoding layers of the SEQ2SEQ model

# Creating the Encoder RNN (main layer):
  
# it will be a stacked LSTM with dropout (to improve the accuracy)
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
  # rnn_inputs are the model inputs,
  # rnn_size is the # of input tensors of the encoder rnn layer
  # keep_prob is for the dropout regularization
  # sequence_length is the size of questions inside each batch
  lstm = tf.keras.layers.LSTMCell(rnn_size)
  lstm_dropout = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob = keep_prob)   
  # apply dropout (20% of neurons' weights are not updated)
  
  # create the encoder cell 
  # (StackedRNNCells: takes lstm with droput as argument * # of layers)
  # _, this variable is the encoder output
  encoder_cell = tf.keras.layers.StackedRNNCells([lstm_dropout] * num_layers)
  _, encoder_state = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                               cell_bw = encoder_cell,
                                                               sequence_length = sequence_length,
                                                               inputs = rnn_inputs,
                                                               dtype = tf.float32)
  # the bidirectional dynamic rnn function returns 2 elements,
  # the encoder state is the 2nd element,
  # so we use _, encoder_state = ...
  # to specify that only we only need the 2nd element returned
  # this function creates a dynamic version of a bidirectional rnn
  # (not a simple rnn), and this step is important to make the chatbot powerful
  # it takes the input, and builds an independent forward and backward rnns
  # but have to be careful: make sure that input size of the forward cell and the backward cell must match
  # the argument makes sure that we have both directions but on the same encoder cell
  # float32 to make sure we are dealing with floats
  
  # the output of this encoder is the state (not the cell: part of the loop in the rnn)
  return encoder_state

  # this is considered the 1st pilar of the seq2seq model
  




# now the Decoding RNN Layer:

# wont be simple, need to do cross validation to separate some training data, and some cross validated data
# cross validation is very important to reduce overfitting and improve accuracy on new observations
# so, start by separating training data from cross validated data
# then, we will be ready to create our decoder rnn layer




# 1. Decoding the training set
# 2. Decoding the validation set
# 3. create the decoder rnn layer


# 1. Decoding the training set
# the decoder in taking in the encoder_state as part of the input
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
  # an embedding is a mapping from words to vectors of real numbers
  # embeddings are important in seq2seq models (as inputs for the decoder)
  # decoding scope is an object of a scope class variable_scope: an advanced data structure that will wrap the tensorflow variables
  # output_function is a function used to return the decoder output in the end
  attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
  # batch_size is # of lines, 1 is # of columns, decoder_cell.output_size is # of elements in 3rd axis
  
  # now, we want to prepare the keys, values, score functions, and construct functions for the attention 
  # we want to preprocess the training data to prepare it to
  # the attention process
  # all this, we can get using a very useful function belonging to the seq2se2 submodule from contrib module: prepare_attention
  attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_options = 'bahdanau', num_units = decoder_cell.output_size)
  # attention keys to be compared with the target states
  # attention values used to construct the cotext vectors.
  # the context is returned by the encoder and is used by the decoder
  # as the 1st element of the decoding
  # the score function is used to compute the similarities b/w keys and target states
  # the attention construct function is a function used to build the attention states
  
  # next step is to get the training decoder function which will do the decoding of the training set
  training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            name = "attn_dec_train")
  decoder_output, _, _ = tf.compat.v1.nn.dynamic_rnn(decoder_cell,
                                                     training_decoder_function,
                                                     decoder_embedded_input,
                                                     sequence_length,
                                                     scope = decoding_scope)
  # we only need the 1st element this function returns (the decoder output)                                                                          
  # the other 2 variables returned are: decoder_final_state, decoder_final_context_state
  
  
  # final step is to apply a final dropout to the decoder output
  decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
  
  return output_function(decoder_output_dropout)




# now decoding of the test/validation set (similar to training set)
# 10% of training set will be the validation set to test the model
# so we reduce overfitting and improve accruacy

# here we wont use attention_decoder_fn_train function
# we will use attention_decoder_fn_inference function
# infer means: to deduce logically (based on what it learned)


# the only difference b/w new observations and training data, is that the new data wont be back propagated,
# but we still need attention bcs its part of the power of prediction (with forward propagation)
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
  # there are 4 new arguments bcs of the new function we will use (inference function)
  attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
  attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_options = 'bahdanau', num_units = decoder_cell.output_size)
  test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                            encoder_state[0],
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            decoder_embeddings_matrix,
                                                                            sos_id,
                                                                            eos_id,
                                                                            maximum_length,
                                                                            num_words,
                                                                            name = "attn_dec_inf")
  # name: is the name of the scope (training or inference mode)
  test_predictions, _, _ = tf.compat.v1.nn.dynamic_rnn(decoder_cell,
                                                       test_decoder_function,
                                                       scope = decoding_scope)

  # dropout is only used in the training to reduce overfitting and improve accuracy 
  return test_predictions





# now the Decoder RNN:

# the word2int dictionary will be our answerswords2int dictionary that we made before
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
  with tf.compat.v1.variable_scope("decoding") as decoding_scope:
    lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    decoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_dropout] * num_layers)
    # the decoder cell is what contains the stacked lstm layers (with dropout applied to each layer)
    
    # now we need to initialize some weights associated with the
    # neurons in the fully connected layer (last layer) inside our decoder
    weights = tf.compat.v1.truncated_normal_initializer(stddev = 0.1)
    # this initializer will generate a truncated normal distribution of the weights
    biases = tf.zeros_initializer()
    # we have our weights and biases for the fully connected layer
    
    # now, we have to make the fully connected layer (last layer for the rnn (for decoder rnn))
    output_function = lambda x: tf.compat.v1.layers.dense(x, 
                                                          num_words,
                                                          scope = decoding_scope,
                                                          weights_initalizer = weights,
                                                          bias_initializer = biases)
  
    # the features of this fully connected layer come from the stacked lstm layers
    # and will return the final score, then using a softmax we will do a final prediction (final answer)


    # now, we want to get our training predictions (from: decode_training_set)
    training_predictions = decode_training_set(encoder_state,
                                               decoder_cell,
                                               decoder_embedded_input,
                                               sequence_length,
                                               decoding_scope,
                                               output_function,
                                               keep_prob,
                                               batch_size)
    
    # now, the test predictions (which is 10% of the training set)
    
    decoding_scope.reuse_variables()  # to reuse the variables introduced in "this" decoding scope
    test_predictions = decode_test_set(encoder_state,
                                       decoder_cell,
                                       decoder_embeddings_matrix,
                                       word2int['<SOS>'],
                                       word2int['<EOS>'],
                                       sequence_length - 1,
                                       num_words,
                                       decoding_scope,
                                       output_function,
                                       keep_prob,
                                       batch_size)
  
  return training_predictions, test_predictions






# now, Buiding the SEQ2SEQ Model using all the tools we created
# this is the brain of the chatbot
# this function is supposed to return the training predictions and the test predictions
# we are essembling here the encoder and decoder
# the encoder which returns the encoder states (taken by the decoder),
# and the decoder which returns the training and test predictions

# but before we get the encoder states, we need the encoder_embedded_inputs
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
  encoder_embedded_input = tf.keras.layers.Embedding(answers_num_words + 1,
                                                     encoder_embedding_size,
                                                     embeddings_initializer = tf.random_uniform_initializer(0, 1))
  # embed sequence will return the embedded inputs for the encoder

  encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
  preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
  decoder_embeddings_matrix = tf.Variable(tf.random_uniform_initializer([questions_num_words + 1, decoder_embedding_size], 0, 1))
  # this intializes the decoder embeddings matrix with random numbers (0, 1) using uniform distribution
  # use this matrix to get the decoder embedded input
  decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
  training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                       decoder_embeddings_matrix,
                                                       encoder_state,
                                                       questions_num_words,
                                                       sequence_length,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int,
                                                       keep_prob,
                                                       batch_size)


  return training_predictions, test_predictions




# now, we move to training the seq2seq model !!

# setting the hyperparameters (used during training)
# can be changed to experiment with them and optimize training
epochs = 100
batch_size = 64                  # (if its slow, try 128... faster)
rnn_size = 512
num_layers = 3                   # layers inside the encoder rnn and decoder rnn
encoder_embedding_size = 512     # number of columns in the embeddings matrix (how many columns we want for the embeddings values)
# in this matrix, each column corresponds to each token in the whole corpus of questions
decoder_embedding_size = 512
learning_rate = 0.01             # this is tricky to choose
learning_rate_decay = 0.9        # 90%   reduces the learning rate over iterations
min_learning_rate = 0.0001       # should not go below this learning rate
keep_probability = 0.5                  # 50% most recommended



# Defining a session in tensorflow, where all the tf training will be done
# reset the graph 1st, to ensure the graph is ready for the training
# (reset whenever opening a tf session)
tf.compat.v1.reset_default_graph()

# create an object of the session
session = tf.compat.v1.InteractiveSession()


# Loading the inputs of the model
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.compat.v1.placeholder_with_default(25, None, name = 'sequence_length')
# sets a default value to produce when the output is not fed into the rnn
# 25 is the max length of the sequences of questions and answers (not morethan 25 words)

#     (all these arguments that will be used in the seq2seq model function)


# Getting the shape of the inputs tensor
# (an argument to be used in the ones() function in tensorflow)
# the dimension of this tensor is gonna be the input shape
input_shape = tf.shape(inputs)
# takes an argument of a tensor, return the shape of the tensor



# now, getting the training and test predictions (using the seq2seq2 model function)
# (when feeding the model with the inputs loaded before)
# reverse function reverses the dimensions of the tensor (inputs need to be reshaped (not in the right shape)) (-1 to reverse)
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoder_embedding_size,  # matrix size
                                                       decoder_embedding_size,  # matrix size
                                                       rnn_size,                # 512
                                                       num_layers,              # 3 layers in the cells (encoder and decoder)
                                                       questionswords2int)      # our dictionary






# setting up the Loss Error, the Optimizer, and Gradient Clipping

# Gradient clipping are operations to ensure no exploding or vanishing gradient happens
# it will cap the gradient b/w a min and max value

# error based on weighted cross entropy, which is the best when dealing with sequences
# optimizer is ADAM, best for SGD

# define a scope for them
with tf.name_scope("optimization"):
  loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                targets,
                                                tf.ones([input_shape[0], sequence_length]))
  
  # now the optimizer, and gradient clipping, then linking gradient clipping to optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate)
  gradients = optimizer.compute_gradients(loss_error)
  clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
  # if grad_tensor is existing ..
  # we want to clip each one of our grad tensors
  # to keep track of each gradient, we form a couple (clip, variable)
  # so this clipping will be done for all the grad_tensors for the gradient in the graph
  optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)



# Padding the sequences with <PAD> token
# bcs all the sentence sequences must have the same length
# question: [ 'who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD> ]
# answer: [ <SOS>, 'I', 'am', 'a', 'bot', '.', <EOS> ]

def apply_padding(batch_of_sequences, word2int): 
# word2int is question or answer dictionary
  max_sequence_length = max ([len(sequence) for sequence in batch_of_sequences])
  # complete other sequences with <PAD> tokens
  return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
  # we need the unique integer id for <PAD> from word2int



# Splitting the data into batches of questions and answers
# then split the questions and answers into training and validation data
# will do cross validation to keep track of the predictive power

# now making the batches
# (inputs are questions and targets are answers)

def split_into_batches(questions, answers, batch_size):
  for batch_index in range(0, len(questions) // batch_size):
    # // gives us an integer (# batches in questions)
    start_index = batch_index * batch_size
    # when index is 0, its the 1st batch
    # when index is 1, its the 2nd batch
    # now add questions and answers in the batch
    questions_in_batch = questions[start_index : start_index + batch_size]
    answers_in_batch = answers[start_index : start_index + batch_size]
    padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
    padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
    # we have returned in the padding function a "list" of sequences
    # but we need a numpy array, bcs in tensorflow we need a numpy array
    
    # we wil use yield instead of returns (its a python method)
    # its better when dealing with sequences
    yield padded_questions_in_batch, padded_answers_in_batch



# now we split questions and answers into training and validation data
# using cross validation (15% of dataset for validation)
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]


# Training:
batch_index_training_loss = 100 # it will check the training loss every 100 batches
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = 0 
# bcs we will use the "early stopping teqnique", to see if we managed to reach the best loss
early_stopping_check = 0 # number of checks, once it reaches a certain value, stop training
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"  # to save the weights and load them whenever we wanna chat with the trained chatbot (crutial)
session.run(tf.global_variables_initializer())    # initialize all global variables in the session
for epoch in range(1, epochs + 1):   # start the huge loop
  for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
    # use enumerate when the function in it (split_into_batches) returns several elements
    starting_time = time.time()  # measure the time for training each time (get difference b/w start and end time to get time of batch)
    _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                          targets: padded_answers_in_batch,
                                                                                          lr: learning_rate
                                                                                          sequence_length: padded_answers_in_batch.shape[1],
                                                                                          keep_prob: keep_probability})
    total_training_loss_error += batch_training_loss_error
    ending_time = time.time()  
    batch_time = ending_time - starting_time
    
    # now, get the training loss error for every 100 batches
    # check if batch_index is divisible by the training loss error
    if batch_index % batch_index_check_training_loss == 0:
      # that means that it reaches 100 batches (or 200, or 300)
      # if thats the case, print the epoch, the batch, the average training loss for 100 batches, and training time for these 100 batches
      print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,      # epoch reached
                                                                                                                           epochs,     # total number of batches
                                                                                                                           batch_index,
                                                                                                                           len(training_questions) // batch_size,  # total # of batches
                                                                                                                           total_training_loss_error / batch_index_check_training_loss,
                                                                                                                           int(batch_time * batch_index_check_training_loss)))   # training time (as integer)
      # 3 figures over the total # of epochs
      # 4 figures over total # of batches
      # 6.3 means 6 figures and 3 decimals for training loss error
      # d means integer for training time
      
      total_training_loss_error = 0  # reset after every 100 batches
      
    # now the validation loss error for validation batch
    if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
      total_validation_loss_error = 0
      starting_time = time.time()
      
      for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
        # here, we are testing the model on new observation
        # so, we dont need an optimizer, since its only used to calculate the gradients (for training)
        batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                               targets: padded_answers_in_batch,
                                                               lr: learning_rate,
                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                               keep_prob: 1})  # bcs the neurons have to always be present (50% is only to improve training)
        total_validation_loss_error += batch_validation_loss_error
      ending_time = time.time()
      batch_time = ending_time - starting_time
      # 
      average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)  # total error / number of batches in validation set
      print('Validation Loss Error: {:6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
      
      
      # apply some decay to the learning rate
      learning_rate *= learning_rate_decay
      
      # we should have a minimum learning rate
      if learning_rate < min_learning_rate:
        # we set it to the min learning rate
        learning_rate = min_learning_rate
      
      # now apply early stopping
      list_validation_loss_error.append(average_validation_loss_error)
      if average_validation_loss_error <= min(list_validation_loss_error):
        print('I speak better now!!')
        early_stopping_check = 0  # whenever we find an improvement in the validation los error
        saver = tf.train.Saver()
        saver.save(session, checkpoint)
      else:
        # if we dont find a validation loss error less than what we have so far
        print("Sorry, I do not speak better, I ned to practice more")
        early_stopping_check += 1
        if early_stopping_check() == early_stopping_stop:
          break
    
    if early_stopping_check() == early_stopping_stop:
      print("my apologies, I cannot speak better anymore. This is my best")
      break
print("Game Over")   



# Loading thhe weights and Running the session
# (after training for a while and saving the weights)
checkpoint = "./chatbots_weights.ckpt"  # contains the path of the weights
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
# connect the loaded weights to the session through the Saver class
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# next, we make a function to convert the sequences and strings
# in questions to lists of encoding integers
def convert_string2int(question, word2int):
  question = clean_text(question)
  return [word2int.get(word, word2int['<OUT>']) for word in question.split()]

# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = ' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
