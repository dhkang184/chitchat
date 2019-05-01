# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:40:00 2017

@author: dhyun
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:32:44 2017

@author: dhyun
"""

# -*- coding: utf-8 -*-

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
    ckpts_dir = os.path.join(dirname , 'checkpoints/10000')
    tb_logs_dir = os.path.join(dirname, 'logs')
    pkl_data_file = os.path.join(dirname, 'metadata_j.pkl') 
    input_file = os.path.join(dirname,'vis_q.npy')    
    target_file = os.path.join(dirname,'vis_a.npy')    

    llprint("Loading Data ... ")

    """
    ### chek point use? unuse?
    """

    pkl_data = pickle.load(open(pkl_data_file, 'rb'))
#    from_checkpoint ='/home/dhyun/PROJECT/DNC_lstm_original/tasks/checkpoints/ubuntu/step-200000'
    from_checkpoint = None
    iterations = 1000000
    start_step = 0 ##woo
    
    ###################lexicon_dict making
    ##################3
    lexicon_dict = pkl_data['w2idx']  ## dictionay making

    dncinput = np.load(input_file)

    dnc_sentence = []
    for l in dncinput:
        input_num = []
        for x in l:
            if x == 0:
                break  
            input_num.append(x)
        dnc_sentence.append(np.array(input_num, dtype=np.int32))
    dncinput = dnc_sentence
    inv_dictionary = idx2w = pkl_data['idx2w']

    llprint("Done!\n")
    
    batch_size = 1
    input_size = len(lexicon_dict)
    output_size = 256 ##autoencoder LSTM hidden unit dimension
    sequence_max_length = 100
    word_space_size = len(lexicon_dict)
    words_count = 256
    word_size = 64
    read_heads = 4

    learning_rate = 1e-3
    momentum = 0.9

    
    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=', 'start='])
 
    hidden_size = 256
    mlp_input = output_size
    
    
    
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

            #optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            #summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

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
            
            ##########################
            ###decoder part
            ##########################
            dec_step =31
            #b = np.load(os.path.join(dirname,'enc.npy'))

            sentence_list = np.load(target_file)
            dec_num = []
            dec_word_dic = pkl_data['w2idx']
            dec_word_dic['<go>'] = len(dec_word_dic)
            dec_word_dic['<eos>'] = len(dec_word_dic)
            dec_values = dec_word_dic.values()
            dec_keys = dec_word_dic.keys()
            dec_dic_rev = dict(zip(dec_values, dec_keys))
            dec_input = []
            dec_tag = []
            for l in sentence_list:
                dec_num = [dec_word_dic['<go>']]
                for x in l:
                    if x == 0:
                        dec_num.append(dec_word_dic['<eos>'])
                        break
                    dec_num.append(x)    
                dec_input.append(np.array(dec_num[:-1], dtype=np.int32))
                dec_tag.append(np.array(dec_num[1:], dtype=np.int32))
            

            output, memory_state = ncomputer.get_outputs()   
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
            
            
            with tf.variable_scope('DEC'):
                outputs, state = tf.nn.dynamic_rnn(dec_cell,
                                                   inputs = emb ,                  # sentence -- (1,time_step,embedding_size)
                                                   initial_state=enc_state)
                                                   
            
            outputs_reshape = tf.reshape(outputs, [-1, hidden_size])
            
            with tf.variable_scope('logit'):
                W_logit = tf.get_variable('W_logit', [hidden_size, dec_one_hot])
                b_logit = tf.get_variable('b_logit', [dec_one_hot])
                dec_logit = tf.matmul(outputs_reshape, W_logit) + b_logit
    
            target_onehot = tf.one_hot(tf.squeeze(dec_target), dec_one_hot)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dec_logit, labels=target_onehot )) 
            dec_mask = np.zeros([len(dec_input),dec_step], dtype = np.float32)
            for ix in range(len(sentence_list)):
                dec_mask[ix, :len(dec_input[ix])-1] = 1
            
            
            ##############WWOOOOWOWOWOWOWOWOO
            

            summeries = []
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate ) 
            summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)
            
            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
            for (grad, var) in gradients:
                if grad is not None:
                    summeries.append(tf.summary.histogram(var.name + '/grad', grad))

            apply_gradients = optimizer.apply_gradients(gradients)

            summeries.append(tf.summary.scalar("Loss", loss))
            

            summerize_op = tf.summary.merge(summeries)
            no_summerize = tf.no_op()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            """
            ##세션 시작########################################################
            """
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1

            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0
            dec_results = []
            enc_inputs = []
            ans_results= []
            output_vec = []
            #for i in range(10):
            len_in =  len(dncinput)
            print "input line numbers: %d" %len_in
            for i in xrange(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    
                    decinput_index = int(i%len_in)
                    if i%len_in ==0:
                        print "epoch : %s" %(i/len_in)
                    ncomputer.memory_state = ncomputer.memory.init_memory()
#                    input_data, seq_len = DNC_input_pre(dncinput[decinput_index],word_space_size)   
                    ### mode 1: mood / 2: weather/ 3: love/4: hobbies/ 5:greeting
                    input_data, seq_len = mode_pre(dncinput[decinput_index] ,word_space_size)
                    #input_data, seq_len = mode_pre(dncinput[decinput_index][1:], int(dncinput[decinput_index][0]) ,word_space_size)
                    decinput = dec_input[decinput_index]
                    dectarget = dec_tag[decinput_index]
                    decmask = dec_mask[decinput_index]
                    summerize = (i % (100) == 0)
                    take_checkpoint = (i != 0) and ((i % end == 0) or (i % 20000 == 0))

                    outputvec,decoderre, loss_value, _, summary, memory = session.run([
                        output,
                        dec_logit,
                        loss,
                        apply_gradients,
                        summerize_op if summerize else no_summerize, memory_state
                    ], feed_dict={
                        ncomputer.input_data: input_data,
                        ncomputer.sequence_length: seq_len,
                        dec_in: np.expand_dims(np.expand_dims(decinput, axis=1),axis=0),
                        dec_target: np.expand_dims(np.expand_dims(dectarget, axis=1),axis=0),
                        mask: np.expand_dims(decmask, axis=0)
                    })

                    last_100_losses.append(loss_value)
                    summerizer.add_summary(summary, i)

                    

                    if summerize:
                        llprint("\n\tAvg. LOSS: %.7f\n" % (np.mean(last_100_losses)))

                        input_sent = ' '.join([inv_dictionary[zz] for zz in input_data[0].argmax(axis =1)[2:]])
                        target_sent = ' '.join([dec_dic_rev[yy] for yy in dectarget])
                        generated_sent = ' '.join([dec_dic_rev[xxx] for xxx in decoderre.argmax(axis=1)])

                        print "input_sentence:" + input_sent
                        print "target_sentence:" + target_sent
                        print "pred_sentence :" + generated_sent
                        #print "mode num:" +str(dncinput[decinput_index][0])
                        #print decoderre[0].argmax(axis=1)

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print "\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time)
                        print "\tApprox. time to completion: %.2f hours" % (estimated_time)
                        #print (memory['read_vectors'])
                    if take_checkpoint:
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                        llprint("Done!\n")
                    
                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-mode_last' )
                    llprint("Done!\n")
                    sys.exit(0)
