import argparse
import csv
import pandas as pd
import ast
#from gensim.models.keyedvectors import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec
import json
import os
import math
import numpy as np
#from tsp import get_distance_matrix


## Transforms data from experiments into form useful for cmr model
def gather_args():
    parser = argparse.ArgumentParser(description='Convert trial data from experiments to a form useful for the cmr model.')
    parser.add_argument('--wordpool', '-w', action='store', default='wordpool.csv', type=str, help='Location of wordpool from which to generate output.')
    parser.add_argument('--exclude', '-ex', action='store_true', default=True, help='Boolean if want to exclude trials that the subject id not pass the size judgement task.')
    parser.add_argument('--output', '-o', action='store', default='cmrdata', type=str, help='Output folder to write the output to.')
    parser.add_argument('--input', '-i', action='store', default='cleantrialdata.csv', type=str, help='Input file containing the trial data which you would like to convert to cmr.')
    parser.add_argument('--experiment', '-e', action='store', default=None, help='If you would like to confine your data to a small subset of the data present in your input file you can do so by putting in the experiment tag to look for.')
    parser.add_argument('--text', '-t', action='store_true', help='By default all of the data will be stored by using the indexes of the words in the list in the wordpool. If you would like to save the data with the actual words themselves use this flag instead.')
    parser.add_argument('--refresh', '-r', action='store_true', help='If flag is set, then the lsa and log probability files will be refreshed.')
    
    args = parser.parse_args()
    return args

######################################################################################
######################################################################################
def get_all_recall_data(input_file, exclude, exp):
    print('Did you remember to clean the trial data first using psiturkdatacleaner.py?')
    
    trial_data = pd.read_csv(input_file)
    if exp is not None: trial_data = trial_data[trial_data['exp'] == exp]
        
    recall_data = trial_data[trial_data['type'].str.contains('recall')]
    reminder_data = trial_data[trial_data['type'].str.contains('reminder')]
    
    if exclude: #take out subjects that did not pass at least 8/10 size judgement tasks
        fails = 0
        results = []
        exclusions = []
        for i in range(recall_data.shape[0]):
            if (len(results) == 10):
                if fails > 2: exclusions.append(recall_data.iloc[i-1]['uniqueid'])
                results = []
                fails = 0
            result = recall_data.iloc[i]['correct_sizes']
            results.append(result)
            if (str(result) == 'False'): fails += 1
                
        for i in range(len(exclusions)):
            subj = str(exclusions[i])
            recall_data = recall_data[~recall_data['uniqueid'].str.match(subj)]
            reminder_data = reminder_data[~reminder_data['uniqueid'].str.match(subj)]
        
    nsubjects = recall_data.shape[0]
    
    return recall_data, reminder_data, nsubjects


######################################################################################
######################################################################################               
def write_standard_pres_file(recall_data, words, cats, output, text):
    locs = []
    cats_show = []
    for i in range(recall_data.shape[0]):
        shown = recall_data.iloc[i]['shown_list']
        locs_i = []
        cats_i = []
        for word in ast.literal_eval(shown):
            if text:
                locs_i.append(word)
            else:
                locs_i.append(words.index(word) + 1)
            cats_i.append(cats[words.index(word)])
        locs.append(locs_i)
        cats_show.append(cats_i)
  
    with open(os.path.join(output, 'pres.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in locs:
            writer.writerow(row)
        
    with open(os.path.join(output, 'pres_cats.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in cats_show:
            writer.writerow(row)
            
            
def write_standard_recalls(recall_data, words, cats, output, text):
    recs = []
    cats_rec = []
    recs_amt = []
    possible_rmds = []  #actual word
    remaining_list = [] #word index
    remaining_cats = [] #cat index
    last_remaining_list = [] #include last recalled word too
    last_remaining_cats = [] #include last recalled word too
    
    for i in range(recall_data.shape[0]):
        
        # get inital recall
        recall = recall_data.iloc[i]['matching_list']
        recs_i = []; cats_i = [];  amt_i = []
        for word in ast.literal_eval(recall):
            if text: recs_i.append(word)
            else: recs_i.append(words.index(word) + 1)
            cats_i.append(cats[words.index(word)] + 1) 
        while len(recs_i) < 16: 
            recs_i.append(0)
            cats_i.append(0)
        recs.append(recs_i)
        cats_rec.append(cats_i)
        amt_i.append(int(recall_data.iloc[i]['num_correct']))
        recs_amt.append(amt_i)
        
        # get remaining recall
        remaining = recall_data.iloc[i]['missing_list']
        remaining_list_i = []; remaining_cats_i = []
        for word in ast.literal_eval(remaining):
            if text: remaining_list_i.append(word)
            else: remaining_list_i.append(words.index(word) + 1)
            remaining_cats_i.append(words.index(word) + 1)   
        while len(remaining_list_i) < 16:
            remaining_list_i.append(-1)
            remaining_cats_i.append(-1)
        remaining_list.append(remaining_list_i)
        remaining_cats.append(remaining_cats_i)
        
        # get remaining recall, including the last word recalled initiallly
        pres = recall_data.iloc[i]['shown_list']
        initial = recall_data.iloc[i]['matching_list']
        last_remaining_list_i = []; last_remaining_cats_i = []
        for word in ast.literal_eval(pres):
            if word not in initial:
                if text: last_remaining_list_i.append(word)
                else: last_remaining_list_i.append(words.index(word) + 1)
                last_remaining_cats_i.append(words.index(word) + 1)
            elif word == pres[15]:
                if text: last_remaining_list_i.append(word)
                else: last_remaining_list_i.append(words.index(word) + 1)
                last_remaining_cats_i.append(words.index(word) + 1)
        while len(last_remaining_list_i) < 16:
            last_remaining_list_i.append(-1)
            last_remaining_cats_i.append(-1)
        last_remaining_list.append(last_remaining_list_i)
        last_remaining_cats.append(last_remaining_cats_i)        
        
        # get index of given reminder from remaining word list
        possible_rmds_i = []
        missing = recall_data.iloc[i]['missing_list']
        for word in ast.literal_eval(missing):
            possible_rmds_i.append(word)
        if len(possible_rmds_i) < 2: possible_rmds_i = [-1]
        possible_rmds.append(possible_rmds_i)
    
    
    # print to files
    with open(os.path.join(output, 'recs.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in recs:
            writer.writerow(row)  
      
    with open(os.path.join(output, 'recs_cats.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in cats_rec:
            writer.writerow(row)
        
            
def write_standard_subjects(recall_data, output):
    ids = {}
    count = 0
    with open(os.path.join(output, 'subject_id_standard.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for i in range(recall_data.shape[0]):
            if recall_data.iloc[i]['uniqueid'] not in ids.keys():
                ids[recall_data.iloc[i]['uniqueid']] = count
                count += 1
            writer.writerow([ids[recall_data.iloc[i]['uniqueid']]])

            
def write_one_subject_per_trial(ntrials, output):
    with open(os.path.join(output, 'subject_id.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for i in range(0, ntrials):
            indx = str(i)
            writer.writerow([indx])
    
    

######################################################################################
######################################################################################
def write_lsa_file(words, output):
    glove_6b_path = os.path.join('../stimuli/glove.6B', 'glove.6B.100d.txt')
    w2v_glove_6b_path = os.path.join('../stimuli/word2vec', 'glove.6B.100d.txt')
    if not os.path.exists(w2v_glove_6b_path):
        glove2word2vec(glove_6b_path, w2v_glove_6b_path)
    glove = KeyedVectors.load_word2vec_format(w2v_glove_6b_path, binary=False)
    sims = []
    for word1 in words:
        sim = []
        for word2 in words:
            sim.append(1 - glove.distance(word1.lower(), word2.lower()))
        sims.append(sim)
    
    with open(os.path.join(output, 'GloVe.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in sims:
            writer.writerow(row)

def write_log_prob(words, output):
    print('LOADING WIKIPEDIA ASSOCIATIONS')
    freq_dict = {}
    glove_6b_path = os.path.join('..', 'stimuli', 'glove.6B', 'glove.6B.100d.txt')
    w2v_glove_6b_path = os.path.join('..', 'stimuli', 'word2vec', 'glove.6B.100d.txt')

    if not os.path.exists(w2v_glove_6b_path):
        glove2word2vec(glove_6b_path, w2v_glove_6b_path)
    glove = KeyedVectors.load_word2vec_format(w2v_glove_6b_path, binary=False)
  
    with open('../stimuli/enwiki-20190320-words-frequency.txt', 'r') as fp:
        reader = csv.reader(fp, delimiter=' ')
        for row in reader:
             freq_dict[row[0]] = int(row[1])
          
    sorted_d = sorted(freq_dict.items(), key=lambda x: - x[1])
    words_n = []
    for i in range(20000):
        if sorted_d[i][0] in glove.vocab:
            words_n.append(sorted_d[i][0])
            
    dists = np.zeros((len(words), len(words_n)))
    print('GATHERING DISTANCES')
    for i, word in enumerate(words):
        for j, targ in enumerate(words_n):
            dists[i][j] = np.dot(glove[word.lower()], glove[targ.lower()])
        if i % 100 == 0:
            print(i)
    softmax_denom = np.sum(np.exp(dists), axis=1)
    
    # Dot distance between two words in glove
    def dot_distance(worda, wordb):
        return np.dot(glove[worda.lower()], glove[wordb.lower()])
    
    wordpool_dists = get_distance_matrix(words, dot_distance)
    
    # Make distances the negative probability such that the shortest path is the one with the most negative path
    def calc_prob_dist(worda, wordb):
        idxa = words.index(worda.upper())
        curr_dist = wordpool_dists[idxa][words.index(wordb.upper())]
        return np.log(np.exp(curr_dist) / softmax_denom[idxa])
      
    prob_matrix = get_distance_matrix(words, calc_prob_dist)
    
    # Save output for later
    with open(os.path.join(output, 'log_probabilites.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in prob_matrix:
            writer.writerow(row)  

###################################################################################### 
######################################################################################
def main():
    args = gather_args()
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    wordpool = pd.read_csv(args.wordpool, header=None)
    words = wordpool.iloc[:,0].to_list()
    cats = wordpool.iloc[:, 2].to_list()

    # cmr model files
    recall_data, reminder_data, ntrials = get_all_recall_data(args.input, args.exclude, args.experiment)
    write_standard_pres_file(recall_data, words, cats, args.output, args.text)
    write_standard_recalls(recall_data, words, cats, args.output, args.text)
    
    # subject_id file
    write_standard_subjects(recall_data, args.output)
    write_one_subject_per_trial(ntrials, args.output)
    
    # glove and log files
    if args.refresh:
        write_lsa_file(words, args.output)
        write_log_prob(words, args.output)

if __name__ == '__main__':
    main()