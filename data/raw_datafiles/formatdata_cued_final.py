import argparse
import numpy as np
import csv
import os


######################################################################################
def gather_args():
    parser = argparse.ArgumentParser(description='Get converted data from formatdata.py and exluded uncued trials or trials in which the participant requested a cue less than 10 seconds into the initial recall period.')
    parser.add_argument('--output', '-o', action='store', default='cmrdata_cued', type=str, help='Output folder to write to.')
    parser.add_argument('--ntrials', '-n', action='store', default=12, type=int, help='Number of trials per subject.')
    args = parser.parse_args()
    return args


######################################################################################
def get_cued_trials(output_file, ntrials):
    print('Did you remember to sort the data using formatdata.py first?')

    ##################
    ### load files ###
    ##################
    subject_id_standard_all = np.loadtxt('cmrdata_all/subject_id_standard.txt', delimiter=',')
    
    pres_all = np.loadtxt('cmrdata_all/pres.txt', delimiter=',')
    pres_cats_all = np.loadtxt('cmrdata_all/pres_cats.txt', delimiter=',')
    
    recs_all = np.loadtxt('cmrdata_all/recs.txt', delimiter=',')
    recs_cats_all = np.loadtxt('cmrdata_all/recs_cats.txt', delimiter=',')
    recs_amts_all = np.loadtxt('cmrdata_all/recs_amt.txt', delimiter=',')
    
    cue_times_all = np.loadtxt('cmrdata_all/cue_times.txt',delimiter=',')
    pres_remaining_all = np.loadtxt('cmrdata_all/pres_remaining.txt', delimiter=',')
    remaining_withlast_all =  np.loadtxt('cmrdata_all/pres_remaining+last.txt', delimiter=',')
    if ntrials==12: 
        with open('cmrdata_all/rmdr_types.txt', 'r') as file: rmdr_types_all = file.readlines()
    rmdr_all = np.loadtxt('cmrdata_all/rmdr.txt', delimiter=',')
    
    post_rmdr_fromcue_all = np.loadtxt('cmrdata_all/post_rmdr_fromcue.txt', delimiter=',')
    post_rmdr_fromlast_all = np.loadtxt('cmrdata_all/post_rmdr_fromlast.txt', delimiter=',')
    post_rmdr_amts_all = np.loadtxt('cmrdata_all/post_rmdr_amt.txt', delimiter=',')
    first_postcue_recall_times_all = np.loadtxt('cmrdata_all/first_postcue_recall_times.txt', delimiter=',')
    
    
    ###################
    ### sort trials ###
    ###################
    subject_id_standard = []
    
    pres = []
    pres_cats = []
    
    recs = []
    recs_cats = []
    recs_amts = []
    
    cue_times = []
    pres_remaining = []
    pres_remaining_last = []
    rmdr_types = []
    rmdr = []
    
    post_rmdr_fromcue = []
    post_rmdr_fromlast = []
    post_rmdr_amts = []
    first_postcue_recall_times = []
    
    subject_id = []; cnt = 0
    for trial in range(len(pres_all)):
        
        cue_times_i = int(cue_times_all[trial])
        
        # if this was a cued trial, (-1 is uncued trials) and Ss persisted in intial recall for >=10 seconds
        if (post_rmdr_amts_all[trial]>=0) and (cue_times_i>=10000):
            
            
            ### subject ids ###
            subject_id.append(cnt); cnt+= 1
            subject_id_standard.append(int(subject_id_standard_all[trial]))
            
            
            ### presentation data ###
            pres_i = []
            for word in pres_all[trial]: pres_i.append(int(word))
            pres.append(pres_i)
            
            pres_cats_i = []
            for word in pres_cats_all[trial]: pres_cats_i.append(int(word))
                
            
            ### recall session data ###
            recs_i = []
            for word in recs_all[trial]: recs_i.append(int(word))
            recs.append(recs_i)
            
            recs_cats_i = []
            for word in recs_cats_all[trial]: recs_cats_i.append(int(word))
            recs_cats.append(recs_cats_i)
            
            recs_amts_i = int(recs_amts_all[trial])
            recs_amts.append(recs_amts_i)
            
            
            ### cue data ###
            cue_times.append(cue_times_i)
            
            pres_remaining_i = []
            for word in pres_remaining_all[trial]: pres_remaining_i.append(int(word))
            pres_remaining.append(pres_remaining_i)
            
            pres_remaining_last_i = []
            for word in remaining_withlast_all[trial]: pres_remaining_last_i.append(int(word))
            pres_remaining_last.append(pres_remaining_last_i)
            
            if ntrials==12:
                trial_type = rmdr_types_all[trial]
                if 'random' in trial_type: rmdr_types_i = 'random'
                elif 'best' in trial_type: rmdr_types_i = 'best'
                elif 'worst' in trial_type: rmdr_types_i = 'worst'
                else: print("REMINDER TYPE ERROR")
                rmdr_types.append(rmdr_types_i)
            
            rmdr_i = int(rmdr_all[trial])
            rmdr.append(rmdr_i)
            
            
            ### reminder session data ###
            post_rmdr_fromcue_i = []
            for word in post_rmdr_fromcue_all[trial]: post_rmdr_fromcue_i.append(int(word))
            post_rmdr_fromcue.append(post_rmdr_fromcue_i)
            
            post_rmdr_fromlast_i = []
            for word in post_rmdr_fromlast_all[trial]: post_rmdr_fromlast_i.append(int(word))
            post_rmdr_fromlast.append(post_rmdr_fromlast_i)
            
            post_rmdr_amts_i = int(post_rmdr_amts_all[trial])
            post_rmdr_amts.append(post_rmdr_amts_i)
            
            first_postcue_recall_times_i = int(first_postcue_recall_times_all[trial])
            first_postcue_recall_times.append(first_postcue_recall_times_i)
            
            
    ###################    
    ### write files ###
    ###################            
    with open('cmrdata_cued/subject_id.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in subject_id: writer.writerow([row])

    with open('cmrdata_cued/subject_id_standard.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in subject_id_standard: writer.writerow([row])
            
    with open('cmrdata_cued/pres.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in pres: writer.writerow(row)
        
    with open('cmrdata_cued/pres_cats.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in pres_cats: writer.writerow(row)
            

    with open('cmrdata_cued/recs.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in recs: writer.writerow(row)

    with open('cmrdata_cued/recs_cats.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in recs_cats: writer.writerow(row)
                        
    with open('cmrdata_cued/recs_amts.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in recs_amts: writer.writerow([row])        
            
    
    with open('cmrdata_cued/cue_times.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in cue_times: writer.writerow([row])
    
    with open('cmrdata_cued/pres_remaining.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in pres_remaining: writer.writerow(row)
            
    with open('cmrdata_cued/pres_remaining_last.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in pres_remaining_last: writer.writerow(row)
            
    if ntrials==12:
        with open('cmrdata_cued/rmdr_types.txt', 'w',newline='') as fp:
            writer = csv.writer(fp)
            for row in rmdr_types: writer.writerow([row])
            
    with open('cmrdata_cued/rmdr.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in rmdr: writer.writerow([row])
            
            
    with open('cmrdata_cued/post_rmdr_fromcue.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in post_rmdr_fromcue: writer.writerow(row)    
            
    with open('cmrdata_cued/post_rmdr_fromlast.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in post_rmdr_fromlast: writer.writerow(row)            
            
    with open('cmrdata_cued/post_rmdr_amt.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in post_rmdr_amts: writer.writerow([row])
            
    with open('cmrdata_cued/first_postcue_recall_times.txt', 'w',newline='') as fp:
        writer = csv.writer(fp)
        for row in first_postcue_recall_times: writer.writerow([row])            
            
    
######################################################################################
def main():
    args = gather_args()
    if not os.path.isdir(args.output): os.mkdir(args.output)
    get_cued_trials(args.output, args.ntrials)

if __name__ == '__main__': 
    main()

    
    
######################################################################################
####  END OF CODE  ###################################################################
######################################################################################