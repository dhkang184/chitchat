# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 02:03:15 2017

@author: dhyun
"""

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os

from dnc.dnc import DNC
from recurrent_controller import RecurrentController
from nltk.translate.bleu_score import sentence_bleu

def linear( _input, _in_ch, _out_ch, _name ):
    w = tf.get_variable(name='%s_w' %_name, shape=[_in_ch, _out_ch], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name='%s_b' %_name, shape=[_out_ch], dtype = tf.float32,
                        initializer = tf.constant_initializer(0.0))
    
    return tf.nn.bias_add( tf.matmul( _input, w ), b )

def lstm(_current_input, _state, _name):
    dim_hidden = int(_state.get_shape()[-1]/2)    
    weight_matrix = tf.get_variable(_name, [dim_hidden+int(_current_input.get_shape()[-1]), 4*dim_hidden], initializer = tf.contrib.layers.xavier_initializer()) 
    bf = tf.get_variable('lstm_bf_%s' %_name, [1, dim_hidden], initializer = tf.constant_initializer(1.0))        
    bi = tf.get_variable('lstm_bi_%s' %_name, [1, dim_hidden], initializer=tf.constant_initializer(0.0))
    bo = tf.get_variable('lstm_bo_%s' %_name, [1, dim_hidden], initializer=tf.constant_initializer(0.0))
    bc = tf.get_variable('lstm_bc_%s' %_name, [1, dim_hidden], initializer=tf.constant_initializer(0.0))
        
    c, h = tf.split(_state, 2, 1)
    input_matrix = tf.concat([h, _current_input], 1 )
    f, i, o, Ct = tf.split(tf.matmul(input_matrix, weight_matrix), 4, 1)
    f = tf.nn.sigmoid(f + bf)
    i = tf.nn.sigmoid(i + bi)
    o = tf.nn.sigmoid(o + bo)        
    Ct = tf.nn.tanh(Ct + bc)
    new_c = f * c + i * Ct
    new_h = o * tf.nn.tanh(new_c)
    new_state = tf.concat([new_c, new_h], 1)								
        
    return new_h, new_state


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec
    
def DNC_input_pre(input_data, word_space_size):
    
    input_vec = np.array(input_data, dtype = np.int32)
    seq_len = input_vec.shape[0]
    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        seq_len)  

def prepare_sample(sample, target_code, word_space_size):
    input_vec = np.array(sample[0]['inputs'], dtype=np.int32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)
    target_mask = (input_vec == target_code)
    output_vec = np.expand_dims(sample[0]['outputs'],1)
    weights_vec[target_mask] = 1.0
    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        output_vec,
        seq_len,
        np.reshape(weights_vec, (1, -1, 1))
    )

def inv_dict(dictionary):
	return {v:k for k, v in dictionary.iteritems()}

def mode_pre(input_data, word_space_size):
    mode_arr = np.array([], dtype = np.int32)
    mode_input = np.concatenate((mode_arr,np.array(input_data, dtype=np.int32)), axis=0)
    input_vec = np.array([onehot(code, word_space_size) for code in mode_input])
    seq_len = input_vec.shape[0]

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        seq_len)     

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints/twitterNN')
    tb_logs_dir = os.path.join(dirname, 'logs')
    ############### pkl_data_file 경로,ckpt_path (checkpoint path) 두개만 설정해주시면 될거 같습니다.
   
    pkl_data_file = os.path.join(dirname, 'metadata_j.pkl')
    ckpt_path = os.path.join(dirname,'checkpoints/10000/step-mode_last/model.ckpt')
    pkl_data = pickle.load(open(pkl_data_file, 'rb'))   
    
    lexicon_dict = pkl_data['w2idx']
    
    llprint("Loading Data ... ")
    

    inv_dictionary = idx2w = pkl_data['idx2w']

    llprint("Done!\n")
    
    batch_size = 1
    input_size = len(lexicon_dict)
    output_size = 1024 ##autoencoder LSTM hidden unit dimension
    sequence_max_length = 100
    word_space_size = len(lexicon_dict)
    words_count = 256
    word_size = 64
    read_heads = 4
    iterations = 100000
    start_step = 0 
    
    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=', 'start='])
 
    hidden_size = 1024
    mlp_input = output_size
    llprint("Done!\n")

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])
        elif opt[0] == '--start':
            start_step = int(opt[1])

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:
        
            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                RecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size
            )
            
            dec_step =31

            dec_num = []
            dec_word_dic = pkl_data['w2idx']
            dec_word_dic['<go>'] = len(dec_word_dic)
            dec_word_dic['<eos>'] = len(dec_word_dic)
            dec_values = dec_word_dic.values()
            dec_keys = dec_word_dic.keys()
            dec_dic_rev = dict(zip(dec_values, dec_keys))
            dec_input = []
            dec_tag = []


            output, _ = ncomputer.get_outputs()
            

            from tensorflow.contrib import rnn           
            
            enc_state = rnn.LSTMStateTuple( c = tf.expand_dims(output[0,-1,:], axis=0),
                                            h = tf.expand_dims(output[0,-1,:], axis=0) )

            dec_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size)
            dec_one_hot = len(dec_word_dic)
            dec_in = tf.placeholder(tf.int32, shape = [batch_size, None,1])
            dec_target = tf.placeholder(tf.int32, shape = [batch_size, None,1])
            mask = tf.placeholder(tf.float32, shape = [batch_size, None])            

            dec_in_one_hot = tf.squeeze(tf.one_hot(dec_in, dec_one_hot), axis=2)

                        
            embedding_mat = tf.Variable(tf.random_normal([dec_one_hot, hidden_size]))
            emb = tf.squeeze(tf.nn.embedding_lookup(embedding_mat, dec_in), axis = 2)            

            length = 1
            
            generated_words = []               
            
            
            for t in range(dec_step):
                if t==0:
                    dec_output = emb[:,0]
                  
                with tf.variable_scope("DEC", reuse=True if t>0 else None):
                    dec_output, enc_state = dec_cell(dec_output, enc_state, scope='rnn/lstm_cell')
                    
                
                with tf.variable_scope("logit", reuse=True if t>0 else None):
                    W_logit = tf.get_variable('W_logit', [hidden_size, dec_one_hot])
                    b_logit = tf.get_variable('b_logit', [dec_one_hot])
                    logit_words = tf.nn.xw_plus_b(dec_output, W_logit, b_logit)

                max_prob_word = tf.argmax(logit_words, 1)
                generated_words.append(max_prob_word)
                output = logit_words               
                
                dec_newin = tf.expand_dims(tf.expand_dims(max_prob_word, axis=1),axis=0)
                newemb = tf.squeeze(tf.nn.embedding_lookup(embedding_mat, dec_newin), axis = 2)
                dec_output = newemb[:,0]
            

            llprint("Done!\n")

            llprint("Initializing Variables ... ")

            session.run(tf.global_variables_initializer())
            llprint("Done!\n")
            var_list = tf.trainable_variables()
            saver = tf.train.Saver(var_list=var_list)
            print [ v.name for v in tf.trainable_variables() ]

            saver.restore(session, ckpt_path)
            
            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1

            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0

            inputsen = []
            predsen = []
            lenth = 100
            overlap_num = 0
            before_out = []

            for z in range(2000):
                    
                try:
                    dec_input = [dec_word_dic['<go>']]
           
                    
                    x = input("input : ")
                    x = x.replace('?',' ?')
                    x = x.replace('!',' !')
                    x = x.replace('.',' .')
                    x = x.replace(',',' ,')
                    x = x.replace('  ', ' ')
                    x = x.replace("'", "")
                    
                    userinput = x.split(' ')
                    user_num = []
                    
                    
                    for l in userinput[:]:
                        try :
                            num = lexicon_dict[l]
                        except KeyError:
                            num = lexicon_dict['<unk>']
                        user_num.append(num)           
                    
                    input_data, seq_len = mode_pre(np.array(user_num),word_space_size)
                    print len(input_data[0])
                    sentence_words_1 = session.run([
                        generated_words
                    ], feed_dict={
                        dec_in: np.expand_dims(np.expand_dims(dec_input, axis=1),axis=0), 
                        ncomputer.input_data: input_data,
                        ncomputer.sequence_length: seq_len,
                    })
                    
                    sentence_words_1 = np.array(sentence_words_1)
                    input_sent = ' '.join([inv_dictionary[zz] for zz in input_data[0].argmax(axis =1)])
                    generated_sent = ' '.join([dec_dic_rev[xxx] for xxx in sentence_words_1[0,:,0]])
                    pred_sent = ""
                    
                    for i in sentence_words_1[0,:,0]:
                                                
                        if i == dec_word_dic["<eos>"]:
                            break
                        pred_sent += dec_dic_rev[i]+" "                        

                    print "input_sentence:" + input_sent
                    print "pred_sentence :" + pred_sent

                    if pred_sent in before_out :
                        overlap_num += 1
                    else:
                        before_out.append(pred_sent)
                        overlap_num =0
                        
                    if overlap_num == 2:
                        overlap_num =0
                        random_size = np.random.choice(30,1)[0]
                        if random_size == 0:
                            random_size = 30
                        random_index = np.random.choice(6,1)[0]
                        restart_num = np.random.choice(20000, random_size)
                        
                        reinput_data, reseq_len = mode_pre(np.array(restart_num),word_space_size)
                        sentence_words_1 = session.run([
                                                generated_words
                                            ], feed_dict={
                                                dec_in: np.expand_dims(np.expand_dims(dec_input, axis=1),axis=0), 
                                                ncomputer.input_data: reinput_data,
                                                ncomputer.sequence_length: reseq_len,
                                            })

                        sentence_words_1 = np.array(sentence_words_1)
                        input_sent = ' '.join([inv_dictionary[zz] for zz in input_data[0].argmax(axis =1)])
                        generated_sent = ' '.join([dec_dic_rev[xxx] for xxx in sentence_words_1[0,:,0]])
                        pred_sent = ""
                        
                        for i in sentence_words_1[0,:,0]:
                                                    
                            if i == dec_word_dic["<eos>"]:
                                break
                            pred_sent += dec_dic_rev[i]+" "                        
    
                        print "reset_input_sentence:" + input_sent
                        print "reset_pred_sentence :" + pred_sent
                        print " "

                        print ("reset")
                    print overlap_num   

                except KeyboardInterrupt:
                    llprint("\nSaving Checkpoint ... "),
                    llprint("Done!\n")
                    sys.exit(0)
