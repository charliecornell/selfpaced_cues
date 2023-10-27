import argparse
import numpy as np
import pandas as pd
import csv
import ast
import json
import os
import math
import difflib
#from gensim.models.keyedvectors import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec
#from tsp import get_distance_matrix


######################################################################################
def gather_args():

    parser = argparse.ArgumentParser(description='Convert trial data from experiments to a form useful for the cmr model.')
    parser.add_argument('--wordpool', '-w', action='store', default='wordpool.csv', type=str, help='Location of wordpool.')
    parser.add_argument('--exclude', '-ex', action='store_true', default=True, help='Boolean to exclude trials by a criterion.')
    parser.add_argument('--output', '-o', action='store', default='cmrdata_all', type=str, help='Output folder to write to.')
    parser.add_argument('--input', '-i', action='store', default='cleantrialdata.csv', type=str, help='Input file with data.')
    parser.add_argument('--ntrials','-n',action='store',default=None, type=int, help='Number of experimental trials per subject.')
    parser.add_argument('--experiment', '-e', action='store', default=None, help='Exp tag if want to process certain data.')
    parser.add_argument('--refresh', '-r', action='store_true', help='If flag set, refreshes lsa and log probability files.')
    parser.add_argument('--text', '-t', action='store_true', help='By default all of data will be stored by using indices of the words in the wordpool list. If you would like to save the data with the actual words themselves, use this flag instead.')    
    args = parser.parse_args()
    return args


######################################################################################
def get_all_recall_data(input_file, exclude, exp, ntrials):
    
    # setup data
    print('Did you remember to clean the trial data first using psiturkdatacleaner.py?')
    trial_data = pd.read_csv(input_file)
    if exp is not None: trial_data = trial_data[trial_data['exp'] == exp]
        
    # separate recall and reminder data
    recall_data = trial_data[trial_data['type'].str.contains('recall')]
    reminder_data = trial_data[trial_data['type'].str.contains('reminder')]
    
    # take out subjects that did not pass at least 8/10 (or 7/9 or 11/14) size judgement tasks (~80%)
    if exclude: 
        cutoff = 2 if ntrials==10 or ntrials==9 else (3 if ntrials==12 else None)
        fails = 0
        results = []
        exclusions = []
        
        for i in range(recall_data.shape[0]):
            if (len(results) == ntrials):
                if fails > cutoff: exclusions.append(recall_data.iloc[i-1]['uniqueid'])
                results = []; fails = 0
                
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
def write_subjects(recall_data, nsubjects, output):
    
    ids = {}; count = 0
    with open(os.path.join(output, 'subject_id_standard.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for i in range(recall_data.shape[0]):
            if recall_data.iloc[i]['uniqueid'] not in ids.keys():
                ids[recall_data.iloc[i]['uniqueid']] = count
                count += 1
            writer.writerow([ids[recall_data.iloc[i]['uniqueid']]])
            
    with open(os.path.join(output, 'subject_id.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for i in range(nsubjects): writer.writerow([str(i)])


######################################################################################               
def write_standard_pres_file(recall_data, words, cats, output, text):
    
    locs = []
    cats_show = []
    
    for i in range(recall_data.shape[0]):
        
        locs_i = []
        cats_i = []
        shown = recall_data.iloc[i]['shown_list']
        
        for word in ast.literal_eval(shown):
            if text: locs_i.append(word)
            else: locs_i.append(words.index(word) + 1)
            cats_i.append(cats[words.index(word)])
            
        locs.append(locs_i)
        cats_show.append(cats_i)
  
    ### write to files ###
    with open(os.path.join(output, 'pres.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in locs: writer.writerow(row)
        
    with open(os.path.join(output, 'pres_cats.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in cats_show: writer.writerow(row)

            
######################################################################################            
def write_standard_recalls(recall_data, words, cats, output, text):
    
    recs = []
    recs_cats = []
    recs_amts = []
    remaining_lists = []
    last_recalls = []
    remaining_withlast = []
    
    for i in range(recall_data.shape[0]):
        
        ### get initial recall data ###
        recs_i = []
        cats_i = []
        last_recall_i = -1 #assume no words recalled intially to start
        recalls = recall_data.iloc[i]['matching_list']
        recs_amt = int(recall_data.iloc[i]['num_correct'])
        
        cnt=1
        for word in ast.literal_eval(recalls):
            if text: recs_i.append(word)
            else: recs_i.append(words.index(word) + 1)
            cats_i.append(cats[words.index(word)] + 1)
            
            if cnt==recs_amt: last_recall_i = word
            cnt+=1
                
        while len(recs_i) < 16: recs_i.append(0); cats_i.append(0) #buffer with 0s
        recs.append(recs_i)
        recs_cats.append(cats_i)
        recs_amts.append([recs_amt])
        last_recalls.append(last_recall_i)
        
        ### get remaining recall lists ###
        remaining = recall_data.iloc[i]['missing_list']
        remaining_lists_i = []
        
        for word in ast.literal_eval(remaining):
            if text: remaining_lists_i.append(word)
            else: remaining_lists_i.append(words.index(word) + 1)  
                
        while len(remaining_lists_i) < 16: remaining_lists_i.append(0)
        remaining_lists.append(remaining_lists_i)
        
        # and get remaining recall list, including the last recall (for CRP plot) ###
        pres = recall_data.iloc[i]['shown_list']
        remaining_withlast_i = []
        lastfound = False
        
        for studied in ast.literal_eval(pres):
            for retrieved in ast.literal_eval(remaining):
                
                if (not lastfound) and (studied == last_recall_i):
                    if text: remaining_withlast_i.append(studied)
                    else: remaining_withlast_i.append(words.index(studied) + 1)
                    lastfound = True #otherwise will record last recall as many times as # of words in remaining list
                
                elif studied == retrieved:
                    if text: remaining_withlast_i.append(studied)
                    else: remaining_withlast_i.append(words.index(studied) + 1)
                        
        while len(remaining_withlast_i) < 16: remaining_withlast_i.append(0)
        remaining_withlast.append(remaining_withlast_i)
        
    
    ### write to files ###
    with open(os.path.join(output, 'recs.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in recs: writer.writerow(row)  
      
    with open(os.path.join(output, 'recs_cats.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in recs_cats: writer.writerow(row)

    with open(os.path.join(output, 'recs_amt.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in recs_amts: writer.writerow(row)
            
    with open(os.path.join(output, 'pres_remaining.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in remaining_lists: writer.writerow(row)
            
    with open(os.path.join(output, 'pres_remaining+last.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in remaining_withlast: writer.writerow(row)
    
    return last_recalls, remaining_lists
            

######################################################################################
def write_standard_reminders(last_recalls, remaining_lists, reminder_data, words, cats, output, text):
    
    rmdr = []
    recs = []
    recs_cats = []
    recs_amts = []
    recs_fromcue = []
    recs_fromlast = []
    first_postcue_recall_times = []
    
    for i in range(reminder_data.shape[0]):
        
        recs_i = []
        cats_i = []
        recs_fromcue_i = []
        last_rec = last_recalls[i]
        recs_fromlast_i = [words.index(last_recalls[i])+1] if (last_rec!=-1) else [0]

        ### get reminder index ###
        cue = (words.index(reminder_data.iloc[i]['reminder'])+1) if not pd.isnull(reminder_data.iloc[i]['reminder']) else ''
        if cue != '':
            for j in range(len(remaining_lists[i])):
                if remaining_lists[i][j] == cue:
                    rmdr.append(j)             # append the reminder index in the remaining list
                    recs_fromcue_i.append(cue) # and track reminder wordpool value for post-cue CRP plot
        else:
            rmdr.append(-1) #if no reminder could be requested
            recs_fromcue_i.append(-1)
        
        ### get reminder session recall ###
        recall = reminder_data.iloc[i]['matching_list']
        for word in ast.literal_eval(recall):
            if text: 
                recs_i.append(word)
                recs_fromcue_i.append(word)
                recs_fromlast_i.append(word)
            else: 
                recs_i.append(words.index(word) + 1)
                recs_fromcue_i.append(words.index(word) + 1)
                recs_fromlast_i.append(words.index(word) + 1)
            cats_i.append(cats[words.index(word)] + 1)
        
        while len(recs_i) < 16: recs_i.append(0); cats_i.append(0)    # buffer with 0s
        while len(recs_fromcue_i) < 16: recs_fromcue_i.append(0)
        while len(recs_fromlast_i) < 16: recs_fromlast_i.append(0)
        recs.append(recs_i)
        recs_cats.append(cats_i)
        recs_fromcue.append(recs_fromcue_i)
        recs_fromlast.append(recs_fromlast_i)
           
        recs_amts_i = []
        if (pd.isnull(reminder_data.iloc[i]['reminder'])):
            if (not pd.isnull(reminder_data.iloc[i]['id'])): recs_amts_i.append('-2') #condition 2 pilot 4
            else: recs_amts_i.append('-1')
        else:
            if (not pd.isnull(reminder_data.iloc[i]['id'])): recs_amts_i.append('-2') #condition 2 pilot 4
            else: recs_amts_i.append(int(reminder_data.iloc[i]['num_correct']))
        recs_amts.append(recs_amts_i)
        
        ### get reaction time of first post-cue recall ###
        times = ast.literal_eval(reminder_data.iloc[i]['times']) #an array
        
        # OPTION 1: only look at first correct recall
#         if cue != '': #if a cued trial,
#             if recs_i[0] >0: #if a correct item was recalled post-cue:
#                 first_recall = ast.literal_eval(reminder_data.iloc[i]['matching_list'])[0]
#                 all_recalls = ast.literal_eval(reminder_data.iloc[i]['recalled_list'])
#                 idx = all_recalls.index(difflib.get_close_matches(first_recall, all_recalls)[0]) #corrects for misspelled words
#                 first_postcue_recall_times.append(times[idx])
#             else: first_postcue_recall_times.append(-1) #represents no correct recalls on cued trial
#         else: first_postcue_recall_times.append(-2) #represents uncued trial
            
        # OPTION 2: just at the first time a correct or incorrect recall was said, but not being the cue word
#         if cue != '': #if a cued trial,
#             all_recalls = ast.literal_eval(reminder_data.iloc[i]['recalled_list'])
#             recorded = False
#             if len(all_recalls) > 0: #if anything was typed in,
#                 for i in range(len(all_recalls)):
#                     if all_recalls[i] != words[cue-1]: 
#                         first_postcue_recall_times.append(times[0])
#                         recorded=True; break
#                 if not recorded: first_postcue_recall_times.append(-1)  #represents no correct or incorrect recalls on cued trial
#             else: first_postcue_recall_times.append(-1) #represents no correct or incorrect recalls on cued trial
#         else: first_postcue_recall_times.append(-2) #uncued trial
            
        # OPTION 3: look at the first time anything was recalled
        if len(times) > 0: first_postcue_recall_times.append(times[0])
        else: first_postcue_recall_times.append(-1)
                                       
    ### write to files ###
    with open(os.path.join(output, 'rmdr.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in rmdr: writer.writerow([row])
            
    with open(os.path.join(output, 'post_rmdr.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in recs: writer.writerow(row)
    
    with open(os.path.join(output, 'post_rmdr_amt.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in recs_amts:  writer.writerow(row)
            
    with open(os.path.join(output, 'post_rmdr_fromcue.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in recs_fromcue: writer.writerow(row)

    with open(os.path.join(output, 'post_rmdr_fromlast.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in recs_fromlast: writer.writerow(row)
            
    with open(os.path.join(output, 'first_postcue_recall_times.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in first_postcue_recall_times: writer.writerow([row])
    

######################################################################################
def write_timing(recall_data, reminder_data, output):
    
    # cue request time during initial recall session
    cue_times = []
    for i in range(recall_data.shape[0]): 
        cue_times.append(int(recall_data.iloc[i]['used_time']))
        
    with open(os.path.join(output, 'cue_times.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in cue_times: writer.writerow([row])
            

######################################################################################
def write_optdata(recall_data, reminder_data, output):

    run_times = []
    for i in range(reminder_data.shape[0]): 
        run_times.append(int(reminder_data.iloc[i]['display_time']))

    with open(os.path.join(output, 'run_times.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in run_times: writer.writerow([row])

    rmdr_types = []
    for i in range(reminder_data.shape[0]):
        trial_type = str(reminder_data.iloc[i]['type'])
        if 'random' in trial_type: rmdr_types.append('random')
        elif 'best' in trial_type: rmdr_types.append('best')
        elif 'worst' in trial_type: rmdr_types.append('worst')
        else: rmdr_types.append('none')

    with open(os.path.join(output, 'rmdr_types.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in rmdr_types: writer.writerow([row])
        

######################################################################################
def write_lsa_file(words, output):
    
    glove_6b_path = os.path.join('../stimuli/glove.6B', 'glove.6B.100d.txt')
    w2v_glove_6b_path = os.path.join('../stimuli/word2vec', 'glove.6B.100d.txt')
    if not os.path.exists(w2v_glove_6b_path): glove2word2vec(glove_6b_path, w2v_glove_6b_path)
    glove = KeyedVectors.load_word2vec_format(w2v_glove_6b_path, binary=False)
    
    sims = []
    for word1 in words:
        sim = []
        for word2 in words: sim.append(1-glove.distance(word1.lower(), word2.lower()))
        sims.append(sim)
    
    with open(os.path.join(output, 'GloVe.txt'), 'w') as fp:
        writer = csv.writer(fp)
        for row in sims: writer.writerow(row)

            
######################################################################################
def write_log_prob(words, output):
                                       
    print('LOADING WIKIPEDIA ASSOCIATIONS')
    freq_dict = {}
    glove_6b_path = os.path.join('..', 'stimuli', 'glove.6B', 'glove.6B.100d.txt')
    w2v_glove_6b_path = os.path.join('..', 'stimuli', 'word2vec', 'glove.6B.100d.txt')

    if not os.path.exists(w2v_glove_6b_path): glove2word2vec(glove_6b_path, w2v_glove_6b_path)
    glove = KeyedVectors.load_word2vec_format(w2v_glove_6b_path, binary=False)
  
    with open('../stimuli/enwiki-20190320-words-frequency.txt', 'r') as fp:
        reader = csv.reader(fp, delimiter=' ')
        for row in reader: freq_dict[row[0]] = int(row[1])
          
    sorted_d = sorted(freq_dict.items(), key=lambda x: - x[1])
    words_n = []
    for i in range(20000):
        if sorted_d[i][0] in glove.vocab: words_n.append(sorted_d[i][0])
            
    dists = np.zeros((len(words), len(words_n)))
    print('GATHERING DISTANCES')
    for i, word in enumerate(words):
        for j, targ in enumerate(words_n): dists[i][j] = np.dot(glove[word.lower()], glove[targ.lower()])
        if i % 100 == 0: print(i)
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
        for row in prob_matrix: writer.writerow(row)  

             
######################################################################################
def main():
    
    # prep
    args = gather_args()
    if args.ntrials==None: print("INPUT NUMBER OF TRIALS PER SUBJECT (--n #).")
    if not os.path.isdir(args.output): os.mkdir(args.output)
        
    wordpool = pd.read_csv(args.wordpool, header=None)
    words = wordpool.iloc[:,0].to_list()
    cats = wordpool.iloc[:, 2].to_list()
    
    # setup data
    if args.refresh: write_lsa_file(words, args.output); write_log_prob(words, args.output)
    recall_data, reminder_data, nsubjects = get_all_recall_data(args.input, args.exclude, args.experiment, args.ntrials)
    
    # get pres data
    write_subjects(recall_data, nsubjects, args.output)
    write_standard_pres_file(recall_data, words, cats, args.output, args.text)
    
    # get precs and rmdr data
    last_recalls, remaining_lists = write_standard_recalls(recall_data, words, cats, args.output, args.text)
    write_standard_reminders(last_recalls, remaining_lists, reminder_data, words, cats, args.output, args.text)
    write_timing(recall_data, reminder_data, args.output)
    
    # if used CMR_reminders to optimize cues
    if args.ntrials==12: write_optdata(recall_data, reminder_data, args.output)
        

if __name__ == '__main__': 
    main()
    

######################################################################################
####  END OF CODE  ###################################################################
######################################################################################