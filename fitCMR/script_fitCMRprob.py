# for cs cluster runtime
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]  = "1"

# import package
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger # for saving 
from bayes_opt.event import Events # for saving
from bayes_opt.util import load_logs # for loading
from random import random

# project specific
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
sys.path.append('../')
from CMRversions.probCMR import CMR2
from CMRversions.functions import init_functions
Functions = init_functions(CMR2)
functions_CMR = Functions()


###############
#
#   Objective function for fitting CMR to behavioral patterns in free recall dataset
# 
###############                                            
def obj_func_prob(beta_enc,beta_rec,gamma_fc,gamma_cf,s_cf,phi_s,phi_d,epsilon_d,k):  
    
    """Error function that we want to minimize"""
    
    ###############
    #
    #   Load parameters and path
    #
    ###############
    
    ll = 16 # list length
    lag_examine = 4 # lags during CRP
    N = 5 # number of simulations per iteration
    LSA_path = '../data/pilotdata/GloVe.txt'
    data_path = '../data/pilotdata/pres.txt'
    data_rec_path = '../data/pilotdata/recs.txt'
    data_cat_path = '../data/pilotdata/pres_cats.txt'
    subjects_path = '../data/pilotdata/subject_id.txt'

    LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(data_rec_path, delimiter=',')
    data_cat = np.loadtxt(data_cat_path, delimiter=',')

    _,data_recalls,data_sp = functions_CMR.data_recode(data_pres, data_rec)  
    data_spc,data_pfr = functions_CMR.get_spc_pfr(data_sp,ll)
    data_crp = functions_CMR.get_crp(data_sp,lag_examine,ll)
    data_LSAs = functions_CMR.get_semantic(data_pres,data_rec,[1,2,3,4],LSA_mat)


    param_dict = {  
        'beta_enc': beta_enc,      # rate of context drift during encoding
        'beta_rec': beta_rec,      # rate of context drift during recall
        'beta_rec_post': 1.0,      # rate of context drift between lists (i.e., post-recall)

        'gamma_fc': gamma_fc,      # learning rate, feature-to-context
        'gamma_cf': gamma_cf,      # learning rate, context-to-feature
        'scale_fc': 1 - gamma_fc,
        'scale_cf': 1 - gamma_cf,

        's_cf': s_cf,              # scales influence of semantic similarity on M_CF matrix
        's_fc': 0.0,               # scales influence of semantic similarity on M_FC matrix.
                                   # s_fc first implemented in Healey et al. 2016; set to 0.0 for prior papers.
                                
        'phi_s': phi_s,      # primacy parameter
        'phi_d': phi_d,      # primacy parameter

        'epsilon_s': 0.0,          # baseline activiation for stopping probability 
        'epsilon_d': epsilon_d,    # scale parameter for stopping probability 
        
        'k': k,   # luce choice rule scale 
        
        'cue_position': -1,   # no initial cue
        
        'primacy': 0.0,       # specific to optimal cmr
        'enc_rate': 1.0,      # chance of encoding success
    }
    
    RMSEs = []    
    for itr in range(N):
    
        # simulate model
        resp, times,_ = functions_CMR.run_CMR2(recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,params=param_dict, subj_id_path=subjects_path, sep_files=False)

        # recode simulations  
        _,CMR_recalls,CMR_sp = functions_CMR.data_recode(data_pres, resp)
        CMR_spc,CMR_pfr = functions_CMR.get_spc_pfr(CMR_sp,ll)
        CMR_crp = functions_CMR.get_crp(CMR_sp,lag_examine,ll)
        CMR_LSAs = functions_CMR.get_semantic(data_pres,data_rec,[1,2,3,4],LSA_mat)

        # calculate fit
        RMSE = 0
        RMSE += functions_CMR.normed_RMSE(data_spc, CMR_spc)   # include SPC
        RMSE += functions_CMR.normed_RMSE(data_pfr, CMR_pfr)   # include PFR
        RMSE += functions_CMR.normed_RMSE(data_crp, CMR_crp)   # include CRP
        RMSE += functions_CMR.normed_RMSE(data_LSAs, CMR_LSAs) # include semantic
        
        # add to list of the N simulation's RMSE    
        RMSEs.append(RMSE) 

    return -np.mean(RMSEs)


    
def main(): 

    # define filename
    filename = "log0822_pilotscued_v0_1.json" # full dataset
    pbounds = {  
        'beta_enc': (0.5,0.85),  # rate of context drift during encoding
        'beta_rec': (0.75,1),    # rate of context drift during recall
        'gamma_fc': (0,0.5),  # learning rate, feature-to-context
        'gamma_cf': (0,0.5),  # learning rate, context-to-feature
        's_cf': (0.75,1.5),   # scale of semantic associations
        'phi_s': (1.0,5.0),   # primacy parameter - scale
        'phi_d': (1.0,3.0),   # primacy parameter - decay
        'epsilon_d': (1,3),   # stopping probability - scale     
        'k': (4,8),           # luce choice rule
        }
        
    # bounded region of parameter space
    optimizer = BayesianOptimization(f=obj_func_prob, pbounds=pbounds,random_state=1)  
      
    # set up logger
    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)   
    
    random.seed(1)
    optimizer.maximize(
        init_points = 400, # number of initial random evaluations
        acq='poi',
        xi=0.01, # acquistion funct; lower=prefer exploitation(eg:0.01), higher=prefer exploration(eg:0.05)
        random_state = random.randint(1,1000000),
        n_iter = 200, # number of evaluations using bayesian optimization
    )


if __name__ == "__main__": main()



