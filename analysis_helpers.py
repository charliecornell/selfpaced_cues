######################################################
###   Imports for analysis, plotting, simulating   ###
######################################################
import math
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from random import shuffle
from matplotlib import gridspec
from scipy.stats import spearmanr, ttest_rel
from scipy import spatial
import matplotlib.pyplot as plt
import csv

import os, sys
sys.path.append('../')
from CMRversions.probCMR import CMR2
from CMRversions.functions import init_functions
Functions = init_functions(CMR2)
functions_CMR = Functions()

from mycolorpy import colorlist as mcp
cset=mcp.gen_color(cmap="inferno",n=8)
cmain=cset[2]
cset=[cset[2],cset[4],cset[5],cset[6], cset[3]]



#####################################################
###   Behavioral patterns during initial recall   ###    
#####################################################
def get_curves(data_path, data_rec_path, subjects_path, param_dict, ll, lag_examine, name):

    # load data
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(data_rec_path, delimiter=',')
    LSA_mat = np.loadtxt('data/pilotdata/GloVe.txt', delimiter=',', dtype=np.float32)
    
    # get data behavior curves
    _, data_recalls, data_sp = functions_CMR.data_recode(data_pres, data_rec)  
    data_spc,data_pfr = functions_CMR.get_spc_pfr(data_sp, ll)
    data_crp = functions_CMR.get_crp(data_sp, lag_examine, ll)
    data_LSAs= functions_CMR.get_semantic(data_pres, data_rec, [1,2,3,4], LSA_mat)
    
    # run CMR on dataset
    recall_mode = 0
    sep_files = False
    resp, _, _ = functions_CMR.run_CMR2(
        recall_mode, LSA_mat, data_path, data_rec_path, param_dict, sep_files, "", subjects_path)
    _, CMR_recalls, CMR_sp = functions_CMR.data_recode(data_pres, resp)

    # get CMR behavior curves
    CMR_spc, CMR_pfr = functions_CMR.get_spc_pfr(CMR_sp, ll)
    CMR_crp = functions_CMR.get_crp(CMR_sp, lag_examine, ll)
    CMR_LSAs= functions_CMR.get_semantic(data_pres, resp, range(1,lag_examine+1), LSA_mat)
    
    # plot
    plot_compare(data_spc,data_pfr,data_crp,data_LSAs, CMR_spc,CMR_pfr,CMR_crp,CMR_LSAs, ll,'Data','CMR',name)
    
    
#####################################################
def plot_compare(spc1,pfr1,crp1,lsa1, spc2,pfr2,crp2,lsa2, ll,label1,label2, name=''):
    
    # Figure settings
    plt.rc('font',size=18)
    plt.rc('axes',titlesize=20)
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.rc('legend',fontsize=20)
    fig, axs = plt.subplots(figsize=(24,5), nrows=1,ncols=4)
    
    axs[0].axis([-1,ll,0,1])
    axs[0].plot(spc1, '-o', color=cmain, alpha=1)
    axs[0].plot(spc2,'--o', color=cmain, alpha=.65, mfc='none')
    axs[0].set_ylabel('Probability of Recall'); axs[0].set_xlabel('Serial Position')
    axs[0].text(-0.3, 1, 'a', transform=axs[0].transAxes, size=32)

    axs[1].axis([-1,ll,0,0.65])
    axs[1].plot(pfr1, '-o', color=cmain, alpha=1)
    axs[1].plot(pfr2,'--o', color=cmain, alpha=.65, mfc='none')
    axs[1].set_ylabel('Probability of\nFirst Recall'); axs[1].set_xlabel('Serial Position')
    axs[0].text(-0.3, 1, 'b', transform=axs[1].transAxes, size=32)

    axs[2].axis([-1,9,0,0.29])
    axs[2].plot(range(4), crp1[0:4], '-o', color=cmain, alpha=1)
    axs[2].plot(range(4), crp2[0:4],'--o', color=cmain, alpha=.65, mfc='none')
    axs[2].plot([5,6,7,8],crp1[5:9], '-o', color=cmain, alpha=1)
    axs[2].plot([5,6,7,8],crp2[5:9],'--o', color=cmain, alpha=.65, mfc='none')
    axs[2].set_xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    axs[2].set_ylabel('Conditional\nResponse Probability'); axs[2].set_xlabel('Lag')
    axs[0].text(-0.3, 1, 'c', transform=axs[2].transAxes, size=32)

    axs[3].axis([0.5,len(lsa1)+.5,0,0.38])
    axs[3].plot(list(range(1,len(lsa1)+1)),lsa1, '-o', color=cmain, alpha=1)
    axs[3].plot(list(range(1,len(lsa2)+1)),lsa2,'--o', color=cmain, alpha=.65, mfc='none')
    axs[3].set_ylabel('Semantic Similarity'); axs[3].set_xlabel('Lag'); axs[3].legend([label1,label2])
    axs[0].text(-0.3, 1, 'd', transform=axs[3].transAxes, size=32)
    
    plt.subplots_adjust(.125, .2, .9, .9, .5, .5) #output plot positioning
    if name!='': plt.savefig("figs/CornellFig"+name+".pdf", format="pdf", bbox_inches="tight")
    plt.show()



#####################################################
###   Conditional probaiblity context plots   #######    
#####################################################
def get_crp_varyll(recalls_sp,lag_examine,lls): 
    """
    Modified from CMRversions.functions.py to return the conditional response probability 
    when the set of trials it is averaged over have various list lengths.
    # recalls_sp: list of lists of serial positions
    # lag_examine: range of lags to examine
    # lls: list of list lengths; index matches trial in recalls_sp
    """          
    possible_counts = np.zeros(2*lag_examine+1)
    actual_counts = np.zeros(2*lag_examine+1)    
    
    for i in range(len(recalls_sp)):
        ll = lls[i]                  # added line; see why important below
        recallornot = np.zeros(ll)       
        
        for j in range(len(recalls_sp[i]))[:-1]:
            sp1 = recalls_sp[i][j]
            sp2 = recalls_sp[i][j+1]                        
            
            if 0 <= sp1 <100:
                recallornot[sp1] = 1
                if 0 <= sp2 <100:
                    lag = sp2 - sp1                    
                    if np.abs(lag) <= lag_examine: 
                        actual_counts[lag+lag_examine] += 1                                      
                    for k,item in enumerate(recallornot):  #importance is here; don't want average influeced by 0% at that lag 
                        if item==0:                                           # if it could not possibly be reached on a trial
                            lag = k - sp1
                            if np.abs(lag) <= lag_examine: 
                                possible_counts[lag+lag_examine] += 1
    
    crp = [(actual_counts[i]+1)/(possible_counts[i]+1) for i in range(len(actual_counts))]  
    crp[lag_examine] = 0
    
    return crp


#####################################################
def post_cue_plots(data_pres, data_recs, LSA_mat, lag_examine,lls,when,plot, post_dist_data,post_dist_CMRA):
    
    # if lls in data_pres vary, get them
    ll = []
    for i in range(len(data_pres)):
        if lls[i] == 'vary':
                ll_i = []
                for trial in data_pres[i]:
                    thisLL = 16
                    for word in trial: 
                        if word<=0: thisLL -= 1
                    ll_i.append(thisLL)
                ll.append(ll_i)
        else: ll.append(lls[i]) #16
      
    # get crp(s) and LSA(s) probabilities
    ncurves = len(data_recs); data_crps = []; data_LSAs = []
    for i in range(ncurves):
        _,data_recalls,data_sp = functions_CMR.data_recode(data_pres[i], data_recs[i])
        if lls[i]=='vary': data_crp = get_crp_varyll(data_sp, lag_examine, ll[i])
        else: data_crp = functions_CMR.get_crp(data_sp, lag_examine, ll[i])
        data_crps.append(data_crp)
        data_LSA = functions_CMR.get_semantic(data_pres[i], data_recs[i], list(range(1,lag_examine+1)), LSA_mat)
        data_LSAs.append(data_LSA)

    # plot data
    if plot:
        fig = plt.figure(figsize = (24,5))
        plt.rc('ytick',labelsize=20); plt.rc('xtick',labelsize=20); 
        plt.rc('axes',labelsize=22); plt.rc('legend',fontsize=18)
        spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[1,1,1.75], wspace=0.35)

        ax0 = fig.add_subplot(spec[0]) # crp plot
        for i in range(ncurves):
            if i==0: 
                ax0.plot(range(lag_examine), data_crps[i][0:lag_examine],'-o', markersize=8, color=cmain)
                ax0.plot(range(lag_examine+1,2*lag_examine+1), data_crps[i][lag_examine+1:2*lag_examine+1],'-o', markersize=8, color=cmain, label='_nolegend_')
            if i==1: 
                ax0.plot(range(lag_examine), data_crps[i][0:lag_examine],':o', mfc='none', markersize=8, color=cmain, alpha=.8)
                ax0.plot(range(lag_examine+1,2*lag_examine+1), data_crps[i][lag_examine+1:2*lag_examine+1],':o', mfc='none', markersize=8, color=cmain, alpha=.8, label='_nolegend_')
        ax0.set_xticks(range(2*lag_examine+1))
        ax0.set_xticklabels(['-4','-3','-2','-1','0','1','2','3','4'])
        ax0.set_xlabel('Lag')
        ax0.set_yticks([y/100.0 for y in range(5,40,5)])
        ax0.set_ylabel('Conditional\nResponse Probability')
        ax0.axis([-0.75,2*lag_examine+.5,0.05,0.35])
        ax0.text(-0.3, 1, 'a', transform=ax0.transAxes, size=35)

        ax1 = fig.add_subplot(spec[1]) # ss plot
        for i in range(ncurves): 
            if i==0: ax1.plot(list(range(1,lag_examine+1)), data_LSAs[i],'-o', markersize=8, color=cmain)
            if i==1: ax1.plot(list(range(1,lag_examine+1)), data_LSAs[i],':o', mfc='none', markersize=8, color=cmain, alpha=.8)
        ax1.set_xticks(range(1,lag_examine+1))
        ax1.set_xticklabels(['1','2','3','4'])
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Semantic Similarity')
        ax1.set_yticks([x/100.0 for x in range(5,40,5)])
        ax1.axis([0.5,lag_examine+.5,0.05,0.35])
        ax0.text(1.1, 1, 'b', transform=ax0.transAxes, size=35)
        ax1.legend(['Data', 'Model'],loc='upper right')

        ax2 = fig.add_subplot(spec[2]) #frequency distribution
        plt.rcParams['hatch.linewidth'] = 1.5
        xaxis = range(len(post_dist_data))
        ax2.bar(xaxis, post_dist_data, width=1, edgecolor=cset[0], alpha=.4, color=cset[0])
        ax2.bar(xaxis, post_dist_CMRA, width=1, edgecolor=cset[0], alpha=.8, hatch='//',fill=None,lw=1.5)
        ax2.set_xticks(xaxis,fontsize=14); ax2.set_yticks([x/100 for x in range(0,70,10)],fontsize=14)
        ax2.set_xlabel('Recall Gain', fontsize=20); ax2.set_ylabel('Propotion of Trials', fontsize=20)
        ax2.legend(['Data','Model'], fontsize=18)
        ax2.text(2.6, 1, 'c', transform=ax0.transAxes, size=35)
        plt.savefig("figs/CornellFig6.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        return [data_crps], [data_LSAs], fig
    else: return [data_crps], [data_LSAs]



#####################################################
###   Better vs worse reminder analysis   ###########    
#####################################################
def perf_by_type(post_amts_data, nsubjects, ntrials_persub, exp, pids, log=False, printing=False):
   
    # add in skipped trials to analyze by subject
    rmdr = np.loadtxt(exp+'/rmdr_all.txt',delimiter=',')
    time = np.loadtxt(exp+'/cue_times_all.txt',delimiter=',')
    with open(exp+'/rmdr_types.txt', 'r') as file: rem_types = file.readlines()
    
    post_amts_data_all = []; rem_types_all = []; cnt=0
    for i in range(int(nsubjects*ntrials_persub)):    
        if rmdr[i]>=0 and time[i]>=10000:
            post_amts_data_all.append(post_amts_data[cnt])
            rem_types_all.append(rem_types[cnt])
            cnt+=1
        else:
            post_amts_data_all.append(-1)
            rem_types_all.append(-1)
    
    # then analyze by subject trial set
    hum_low = []; hum_upp = []; hum_rnd = []; skipped = []
    
    for subject in range(nsubjects):
        if subject in pids:
            
            hum_low_i = []; hum_upp_i = []; hum_rnd_i = []
            
            for trial in range(ntrials_persub):
                
                index = subject*ntrials_persub + trial
                perf_data = post_amts_data_all[index]
                
                if rem_types_all[index] == 'worst\n': hum_low_i.append(perf_data)
                elif rem_types_all[index] == 'best\n': hum_upp_i.append(perf_data)
                elif rem_types_all[index] == 'random\n':  hum_rnd_i.append(perf_data)
            
            if len(hum_low_i) > 0 and len(hum_upp_i) and len(hum_rnd_i)> 0:
                hum_low.append(np.mean(hum_low_i))
                hum_upp.append(np.mean(hum_upp_i))
                hum_rnd.append(np.mean(hum_rnd_i))
                
            else: 
                skipped.append(subject)
                if printing: print("skipped",subject,"who had",len(hum_low_i),"worst trials and",
                                                             len(hum_rnd_i),"random trials and",
                                                             len(hum_upp_i),"upper trials.")
            
    return hum_low, hum_rnd, hum_upp, skipped


#####################################################
def permutation_test(data, name, test):
    
    perms = 1000
    worst = data[0]; random = data[1]; best = data[2]; 
    diff_BR = []; diff_RW = []; diff_BW = []

    for permutations in range(perms):

        br = best+random; shuffle(br) #pool and permute BEST and RANDOM conditions
        diff_BR.append(np.mean(br[:int(len(br)/2)])-np.mean(br[int(len(br)/2):])) #find difference

        rw = random + worst; shuffle(rw) #pool and permute RANDOM and WORST conditions
        diff_RW.append(np.mean(rw[:int(len(rw)/2)])-np.mean(rw[int(len(rw)/2):])) #find difference

        bw = best + worst; shuffle(bw) #pool and permute BEST and WORST conditions
        diff_BW.append(np.mean(bw[:int(len(bw)/2)])-np.mean(bw[int(len(rw)/2):])) #find difference

    obs_diff_BW = abs(np.mean(data[2]) - np.mean(data[0])); cnt = 0
    for d in range(perms):
        if test=='increasing':
            if diff_BR[d] > 0 and diff_RW[d] > 0 and diff_BW[d] > obs_diff_BW: cnt +=1
        elif test=='decreasing':
            if diff_BR[d] < 0 and diff_RW[d] < 0 and diff_BW[d] > obs_diff_BW: cnt +=1
            
    print(" * Permutation test for",name,"has p =",cnt/perms)
    return cnt/perms


#####################################################
def plot_rmdr_types(data):
    
    # sample and ttest
    x=data[0]; y=data[2]
    t,p=ttest_rel(x,y); sig = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else " n.s. "))

    # effect size and its CI
    u1 = np.mean(x); n1 = len(x); s1 = np.std(x)
    u2 = np.mean(y); n2 = len(y); s2 = np.std(y)
    d = (u1 - u2) / (np.sqrt( ((n1 - 1) * s1**2 + (n2 - 1) * s2**2 ) / (n1 + n2 - 2)))
    std_d = np.sqrt( ((n1+n2)/(n1*n2)) + ((d**2)/(2*(n1+n2))) )
    low_ci_d = d-1.96*std_d; upp_ci_d = d+1.96*std_d
        
    # plot
    rs = ['Worst','Random','Best']
    xlabel = "Cue Conditions"; ylabel = 'Recall Gain'; avg = []; rtype = []; dtype = []
    for x in range(len(data[0])): avg.append(data[0][x]); rtype.append(rs[0]); dtype.append('All')
    for x in range(len(data[1])): avg.append(data[1][x]); rtype.append(rs[1]); dtype.append('All')
    for x in range(len(data[2])): avg.append(data[2][x]); rtype.append(rs[2]); dtype.append('All')
    df = pd.DataFrame({xlabel:rtype, ylabel:avg, "data":dtype})

    sns.set_style('white'); sns.set_context("paper", font_scale=1.4)
    p = sns.FacetGrid(data=df, col="data", height=5, aspect=8/10)
    p.map_dataframe(sns.barplot, x=xlabel, y=ylabel, errorbar=None, capsize=.15, palette="magma", alpha=.8)
    axes = p.axes.flatten()
    axes[0].set_title('',fontsize=16)
    axes[0].errorbar(x=[0,1,2], y=[np.mean(data[j]) for j in range(3)],
                     yerr=[np.std(data[j])/np.sqrt(len(data[j])) for j in range(3)], #within Ss SEM bars
                     color='black', ls='none', lw=2, capsize=5)
    axes[0].text(-0.25, 1, 'a', transform=axes[0].transAxes, size=25)
    plt.savefig("figs/CornellFig7a.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    

    
#####################################################
###   Context change analysis   #####################    
#####################################################
def get_cosine_similarity(net_cs, CMR_sp, reminders, rmdr_idxs_CMR, ntrials, nsubjects, ntrials_persub, exp):
    
    # obtain contexts of all the remaining words and cue word
    remaining_cs = []; cue_cs = []
    for trial in range(ntrials):
        remaining_cs_trial = []
        for i in range(len(net_cs[trial])): #list length (net_cs in presentation order)
            if i == reminders[trial][rmdr_idxs_CMR[trial]]: #if the cue
                c = []
                for j in range(len(net_cs[trial][i])): c.append(net_cs[trial][i][j][0])
                cue_cs.append(c)
            elif i not in CMR_sp[trial]: #not recalled (or the cue from if statement)
                c = []
                for j in range(len(net_cs[trial][i])): c.append(net_cs[trial][i][j][0])
                remaining_cs_trial.append(c)
        remaining_cs.append(remaining_cs_trial)

    # find average cosine similarity of cue context and contexts of each remaining word
    cos_sims = []; cnt=0
    for trial in range(ntrials):
        sim = [] # obtain cosine similarity between cue and each remaining word
        for item in remaining_cs[trial]: sim.append (1 - spatial.distance.cosine(cue_cs[trial], item))
        cos_sims.append(np.mean(sim)) # append average cosine similarity for trial

    # add in skipped trials to analyze by subject
    rmdr = np.loadtxt(exp+'/rmdr_all.txt',delimiter=',')
    time = np.loadtxt(exp+'/cue_times_all.txt',delimiter=',')
    with open(exp+'/rmdr_types.txt', 'r') as file: rem_types = file.readlines()
    cos_sims_all = []; rem_types_all = []; cnt=0
    for i in range(int(nsubjects*ntrials_persub)):    
        if rmdr[i]>=0 and time[i]>=10000: cos_sims_all.append(cos_sims[cnt]); rem_types_all.append(rem_types[cnt]); cnt+=1
        else: cos_sims_all.append(-1); rem_types_all.append(-1)

    # then obtain data by subject trial set
    worst_csim = []; rand_csim = []; best_csim = []
    for subject in range(nsubjects):
        worst_i = []; rand_i = []; best_i = []
        for trial in range(ntrials_persub):
            index = subject*ntrials_persub + trial
            if rem_types_all[index] == 'worst\n': worst_i.append(cos_sims_all[index])
            elif rem_types_all[index] == 'best\n': best_i.append(cos_sims_all[index])
            elif rem_types_all[index] == 'random\n': rand_i.append(cos_sims_all[index])
        if len(worst_i)>0 and len(best_i)>0 and len(rand_i)>0:
            worst_csim.append(np.mean(worst_i)); best_csim.append(np.mean(best_i)); rand_csim.append(np.mean(rand_i))

    return worst_csim, rand_csim, best_csim


#####################################################
def plot_cueconds(data,ylabel,ystep,ymax,ytext,letter):
    
    # sample and ttest
    x=data[0]; y=data[2]
    t,p=ttest_rel(x,y); sig = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else " ns "))

    # effect size and its CI
    u1 = np.mean(x); n1 = len(x); s1 = np.std(x)
    u2 = np.mean(y); n2 = len(y); s2 = np.std(y)
    d = (u1 - u2) / (np.sqrt( ((n1 - 1) * s1**2 + (n2 - 1) * s2**2 ) / (n1 + n2 - 2)))
    std_d = np.sqrt( ((n1+n2)/(n1*n2)) + ((d**2)/(2*(n1+n2))) )
    low_ci_d = d-1.96*std_d; upp_ci_d = d+1.96*std_d

    # output
    print(" * Worst vs Best: t("+str(len(x)-1)+")=%.2f, p=%.3f, d=%.2f, 95CI=[%.2f,%.2f]"%(t,p,d,low_ci_d,upp_ci_d))
        
    # plot
    rs = ['Worst','Random','Best']; avg = []; rtype = []; dtype = []
    xlabel = "Cue Conditions"; ylabel = ylabel
    for x in range(len(data[0])): avg.append(data[0][x]); rtype.append(rs[0]); dtype.append('CMR')
    for x in range(len(data[1])): avg.append(data[1][x]); rtype.append(rs[1]); dtype.append('CMR')
    for x in range(len(data[2])): avg.append(data[2][x]); rtype.append(rs[2]); dtype.append('CMR')
    df = pd.DataFrame({xlabel:rtype, ylabel:avg, "data":dtype})

    sns.set_style('white'); sns.set_context("paper", font_scale=1.4)
    p = sns.FacetGrid(data=df, col="data", height=5, aspect=8/10)
    p.map_dataframe(sns.barplot, x=xlabel, y=ylabel, errorbar=None, palette="magma", alpha=.8)
    axes = p.axes.flatten(); axes[0].set_title("")
    axes[0].set_yticks([x/100 for x in range(0,int(ymax*100+ystep*100),int(ystep*100))])
    axes[0].errorbar(x=[0,1,2], y=[np.mean(data[j]) for j in [0,1,2]],
                         yerr=[np.std(data[j])/np.sqrt(len(data[j])) for j in [0,1,2]], #within Ss SE
                         color='black', ls='none', lw=2, capsize=5)
    axes[0].text(-0.25, 1, letter, transform=axes[0].transAxes, size=23)
    plt.savefig("figs/CornellFig7"+letter+".pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    