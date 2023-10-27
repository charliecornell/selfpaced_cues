import numpy as np
import time
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger # for saving 
from bayes_opt.event import Events # for saving
from bayes_opt.util import load_logs # for loading
import fitCMR.script_fitCMRprob as fitCMR

from probCMR_overrides import CMR2Reminder
from CMRversions.functions import init_functions
Functions = init_functions(CMR2Reminder)

import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
cset=mcp.gen_color(cmap="inferno",n=8)


class FunctionsReminder(Functions):
    
    def run_CMR2_singleSubj(self, recall_mode, pres_sheet, rec_sheet, LSA_mat, params):
        """Run CMR2 for an individual subject / data sheet"""

        # initialize lists to store CMR2 output
        resp_values = []
        support_values = []
        reminders_values = []
        recalls_values = []
        accs_values = []
        pcas_values = []
        pca2_values = []
        net_contexts_values = []
        
        # create CMR2 object
        this_CMR = CMR2Reminder(recall_mode, params, LSA_mat, pres_sheet, rec_sheet)

        # layer LSA cos theta values onto the weight matrices
        this_CMR.create_semantic_structure()

        # run CMR2 for each list
        for i in range(len(this_CMR.pres_list_nos)):
            
            # present new list
            this_CMR.present_list()
            
            # and for analysis only, obtain encoding context for each item in the list
            net_contexts = this_CMR.get_context_space()
            net_contexts_values.append(net_contexts)

            # recall session
            rec_items, support = this_CMR.recall_session()
            resp_values.append(rec_items)
            support_values.append(support)

            # reminder session
            reminders, recalls, accs, pcas, pca2 = this_CMR.reminder_session()
            reminders_values.append(reminders)
            recalls_values.append(recalls)
            accs_values.append(accs)
            pcas_values.append(pcas)
            pca2_values.append(pca2)
            
        return resp_values, support_values, this_CMR.lkh, reminders_values, recalls_values, accs_values, pcas_values, pca2_values, net_contexts_values

    

    def run_CMR2(self, recall_mode, LSA_mat, data_path, rec_path, params, sep_files, filename_stem="", subj_id_path="."):
        """Run CMR2 for all subjects.

        time_values = time for each item since beginning of recall session

        For later zero-padding the output, we will get list length from the width of presented-items matrix. 
        This assumes equal list lengths  across Ss and sessions, unless you are inputting each session
        individually as its own matrix, in which case, list length will update accordingly.

        If all Subjects' data are combined into one big file, as in some files from prior CMR2 papers, then divide data into 
        individual sheets per subject If you want to simulate CMR2 for individual sessions, then you can feed in individual 
        session sheets at a time, rather than full subject presented-item sheets.
        """

        now_test = time.time()

        # set diagonals of LSA matrix to 0.0
        np.fill_diagonal(LSA_mat, 0)

        # initialize lists to store CMR2 output
        resp_vals_allSs = []
        support_vals_allSs = []
        reminders_values_allSs = []
        recalls_values_allSs = []
        accs_values_allSs = []
        pcas_values_allSs = []
        lkh = 0
        pca2_values_allSs = []
        net_contexts_values_allSs = []

        # simulate each subject's responses
        if not sep_files:

            # divide up the data
            subj_presented_data, subj_recalled_data, unique_subj_ids = self.separate_files(data_path, rec_path, subj_id_path)

            # get list length
            listlength = subj_presented_data[0].shape[1]

            # for each subject's data matrix,
            for m, pres_sheet in enumerate(subj_presented_data):
                rec_sheet = subj_recalled_data[m]
                subj_id = unique_subj_ids[m]
                # print('Subject ID is: ' + str(subj_id))

                resp_Subj, support_Subj, lkh_Subj, reminders_values, recalls_values, accs_values, pcas_values, pca2_values, net_contexts_values = self.run_CMR2_singleSubj(recall_mode, pres_sheet, rec_sheet, LSA_mat, params)

                resp_vals_allSs.append(resp_Subj)
                support_vals_allSs.append(support_Subj)
                reminders_values_allSs.extend(reminders_values)
                recalls_values_allSs.extend(recalls_values)
                accs_values_allSs.extend(accs_values)
                pcas_values_allSs.extend(pcas_values)
                lkh += lkh_Subj
                pca2_values_allSs.extend(pca2_values)
                net_contexts_values_allSs.extend(net_contexts_values)
                
        # if files are separate, then read in each file individually
        else:
            # get all the individual data file paths
            indiv_file_paths = glob(data_path + filename_stem + "*.mat")

            # read in the data for each path & stick it in a list of data matrices
            for file_path in indiv_file_paths:

                data_file = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)  # get data
                data_mat = data_file['data'].pres_itemnos  # get presented items

                resp_Subj, support_Subj, lkh_Subj, reminders_values, recalls_values, accs_values, pcas_values, pca2_values, net_contexts_values = self.run_CMR2_singleSubj(recall_mode, data_mat, LSA_mat, params)

                resp_vals_allSs.append(resp_Subj)
                support_vals_allSs.append(support_Subj)
                reminders_values_allSs.extend(reminders_values)
                recalls_values_allSs.extend(recalls_values)
                accs_values_allSs.extend(accs_values)
                pcas_values_allSs.extend(pcas_values)
                lkh += lkh_Subj
                pca2_values_allSs.extend(pca2_values)
                net_contexts_values_allSs.extend(net_contexts_values)
                
            # for later zero-padding the output, get list length from one file.
            data_file = scipy.io.loadmat(indiv_file_paths[0], squeeze_me=True, truct_as_record=False)
            data_mat = data_file['data'].pres_itemnos

            listlength = data_mat.shape[1]

        ##############
        #
        #   Zero-pad the output
        #
        ##############
        
        # If more than one subject, reshape the output into a single, consolidated sheet across all Ss
        if len(resp_vals_allSs) > 0:
            resp_values = [item for submat in resp_vals_allSs for item in submat]
            support_values = [item for submat in support_vals_allSs for item in submat]
        else:
            resp_values = resp_vals_allSs
            support_values = support_vals_allSs

        # set max width for zero-padded response matrix
        maxlen = listlength * 1
        nlists = len(resp_values)

        # initialize zero matrices of desired shape
        resp_mat    = np.zeros((nlists, maxlen))
        support_mat = np.zeros((nlists, maxlen))

        # place output in from the left
        for row_idx, row in enumerate(resp_values):
            resp_mat[row_idx][:len(row)]  = resp_values[row_idx]
            if recall_mode==0: support_mat[row_idx][:len(row)] = support_values[row_idx]
                
        #print('Analyses complete.')
        #print("CMR Time: " + str(time.time() - now_test))
        return resp_mat, support_mat, lkh, reminders_values_allSs, recalls_values_allSs, accs_values_allSs, pcas_values_allSs, pca2_values_allSs, net_contexts_values_allSs
    
    
    
    def load_data(self,data_id):
        
        if data_id==0: # only included cued trials for RANDOM PILOT reminders experiment
            data_path = 'data/pilotdata/pres.txt'
            data_rec_path = 'data/pilotdata/recs.txt'
            data_cat_path = 'data/pilotdata/pres_cat.txt'
            subjects_path = 'data/pilotdata/subject_id.txt' # assume each list is a subject
        
        if data_id==1: # only included cued trials for BEST/WORST PILOT reminders experiment
            data_path = 'data/pilotdata_opt/pres.txt'
            data_rec_path = 'data/pilotdata_opt/recs.txt'
            data_cat_path = 'data/pilotdata_opt/pres_cat.txt'
            subjects_path = 'data/pilotdata_opt/subject_id.txt' # assume each list is a subject
            
        if data_id==2: # only included cued trials for PRE-REG'D reminders experiment
            data_path = 'data/finaldata_opt/pres.txt'
            data_rec_path = 'data/finaldata_opt/recs.txt'
            data_cat_path = 'data/finaldata_opt/pres_cat.txt'
            subjects_path = 'data/finaldata_opt/subject_id.txt' # assume each list is a subject
            
        LSA_path = 'data/pilotdata/GloVe.txt'
        LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)
        
        return LSA_mat, data_path, data_rec_path, data_cat_path, subjects_path
    
    
    
    def model_probCMR(self, N, ll, lag_examine, data_id, subject=-1):  
        """ Error function that we want to minimize by simulatating free recall data
            N = 0: obtain lists of recall, in serial positions
            N = 1: force recall order (to, for example, obtain likelihood given data but not returned now)
            N > 2: plot behavioral data with error bar with N being the number of times in simulations
            ll: list length
            lag_examine: lag used in plotting CRP 
            data_id: determines path to dataset files and parameter fit to use
            subject: when data_id==3, this identifies the individual participant to obtain their model fit
        """
        # set up files
        LSA_mat, data_path, data_rec_path, data_cat_path, subjects_path = self.load_data(data_id)
        data_pres = np.loadtxt(data_path, delimiter=',')
        data_rec = np.loadtxt(data_rec_path, delimiter=',')
        
        ###############
        #
        # set model parameters
        #
        ###############
        
        param_dict = {   #log0822_v1 (pre-registered)
            'beta_enc':  0.8500000000000000,       # rate of context drift during encoding
            'beta_rec':  0.8277075712894073,       # rate of context drift during recall
            'beta_rec_post': 1,                    # rate of context drift between lists (i.e., post-recall)

            'gamma_fc': 0.34118896387005193,        # learning rate, feature-to-context
            'gamma_cf': 0.31550734698573957,       # learning rate, context-to-feature
            'scale_fc': 1 - 0.34118896387005193,
            'scale_cf': 1 - 0.31550734698573957,

            's_cf': 1.406054360092629,      # scales influence of semantic similarity on M_CF matrix
            's_fc': 0.0,                     # scales influence of semantic similarity on M_FC matrix
                                             # s_fc first implemented in Healey etal. 2016; set to 0.0 for prior papers

            'phi_s': 4.371265665679813,      # primacy parameter
            'phi_d': 2.2261816044420493,     # primacy parameter

            'epsilon_s': 0.0,                # baseline activiation for stopping probability 
            'epsilon_d': 1.8458529758549949, # scale parameter for stopping probability 

            'k':  6.1396204443630795,        # scale parameter in luce choice rule during recall

            'primacy': 0.0,        # specific to optimal CMR
            'enc_rate': 1.0,       # specific to optimal CMR
        }
            
        # simulated model based on parameter set and 
        if N==0 or N==1: 
            
            # N=0: use CMR's initial recall to determine reminder session list
            # N=1: use participant's initial recall to determine reminder session list
            
            resp, times, _, reminders, recalls, accs, pcas, pcas2, net_contexts = self.run_CMR2(recall_mode=N, LSA_mat=LSA_mat, 
                data_path=data_path, rec_path=data_rec_path, params=param_dict, subj_id_path=subjects_path, sep_files=False)
            _, CMR_recalls, CMR_sp = self.data_recode(data_pres, resp)
            return CMR_sp, reminders, recalls, accs, pcas, pcas2, net_contexts

        
        