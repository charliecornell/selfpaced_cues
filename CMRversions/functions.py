import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import time

def init_functions(ProbCMR, cmr_version_dir='CMRversions/'):
    class Functions:
        def run_CMR2_singleSubj(self, recall_mode, pres_sheet, rec_sheet, LSA_mat, params):

            """Run CMR2 for an individual subject / data sheet"""

            # init. lists to store CMR2 output
            resp_values = []
            support_values = []

            # create CMR2 object
            this_CMR = ProbCMR(
                recall_mode=recall_mode, params=params,
                LSA_mat=LSA_mat, pres_sheet=pres_sheet, rec_sheet=rec_sheet)

            # layer LSA cos theta values onto the weight matrices
            this_CMR.create_semantic_structure()

            # Run CMR2 for each list
            for i in range(len(this_CMR.pres_list_nos)):
                # present new list
                this_CMR.present_list()

                # recall session
                rec_items_i, support_i = this_CMR.recall_session()

                # append recall responses & times
                resp_values.append(rec_items_i)
                support_values.append(support_i)
            return resp_values, support_values, this_CMR.lkh


        def run_CMR2(self, recall_mode, LSA_mat, data_path, rec_path, params, sep_files,
                     filename_stem="", subj_id_path="."):
            """Run CMR2 for all subjects

            time_values = time for each item since beginning of recall session

            For later zero-padding the output, we will get list length from the
            width of presented-items matrix. This assumes equal list lengths
            across Ss and sessions, unless you are inputting each session
            individually as its own matrix, in which case, list length will
            update accordingly.

            If all Subjects' data are combined into one big file, as in some files
            from prior CMR2 papers, then divide data into individual sheets per subj.

            If you want to simulate CMR2 for individual sessions, then you can
            feed in individual session sheets at a time, rather than full subject
            presented-item sheets.
            """

            now_test = time.time()

            # set diagonals of LSA matrix to 0.0
            np.fill_diagonal(LSA_mat, 0)

            # init. lists to store CMR2 output
            resp_vals_allSs = []
            support_vals_allSs = []
            lkh = 0

            # Simulate each subject's responses.
            if not sep_files:

                # divide up the data
                subj_presented_data, subj_recalled_data, unique_subj_ids = self.separate_files(
                    data_path, rec_path, subj_id_path)

                # get list length
                listlength = subj_presented_data[0].shape[1]

                # for each subject's data matrix,
                for m, pres_sheet in enumerate(subj_presented_data):
                    rec_sheet = subj_recalled_data[m]
                    subj_id = unique_subj_ids[m]
                    # print('Subject ID is: ' + str(subj_id))

                    resp_Subj, support_Subj, lkh_Subj = self.run_CMR2_singleSubj(
                                recall_mode, pres_sheet, rec_sheet, LSA_mat,
                                params)

                    resp_vals_allSs.append(resp_Subj)
                    support_vals_allSs.append(support_Subj)
                    lkh += lkh_Subj
            # If files are separate, then read in each file individually
            else:

                # get all the individual data file paths
                indiv_file_paths = glob(data_path + filename_stem + "*.mat")

                # read in the data for each path & stick it in a list of data matrices
                for file_path in indiv_file_paths:

                    data_file = scipy.io.loadmat(
                        file_path, squeeze_me=True, struct_as_record=False)  # get data
                    data_mat = data_file['data'].pres_itemnos  # get presented items

                    resp_Subj, support_Subj, lkh_Subj = self.run_CMR2_singleSubj(
                        recall_mode, data_mat, LSA_mat,
                        params)

                    resp_vals_allSs.append(resp_Subj)
                    support_vals_allSs.append(support_Subj)
                    lkh += lkh_Subj

                # for later zero-padding the output, get list length from one file.
                data_file = scipy.io.loadmat(indiv_file_paths[0], squeeze_me=True,
                                             struct_as_record=False)
                data_mat = data_file['data'].pres_itemnos

                listlength = data_mat.shape[1]


            ##############
            #
            #   Zero-pad the output
            #
            ##############

            # If more than one subject, reshape the output into a single,
            # consolidated sheet across all Ss
            if len(resp_vals_allSs) > 0:
                resp_values = [item for submat in resp_vals_allSs for item in submat]
                support_values = [item for submat in support_vals_allSs for item in submat]
            else:
                resp_values = resp_vals_allSs
                support_values = support_vals_allSs

            # set max width for zero-padded response matrix
            maxlen = listlength * 1

            nlists = len(resp_values)

            # init. zero matrices of desired shape
            resp_mat  = np.zeros((nlists, maxlen))
            support_mat   = np.zeros((nlists, maxlen))


            # place output in from the left
            for row_idx, row in enumerate(resp_values):

                resp_mat[row_idx][:len(row)]  = resp_values[row_idx]
                support_mat[row_idx][:len(row)]   = support_values[row_idx]


            #print('Analyses complete.')

            #print("CMR Time: " + str(time.time() - now_test))
            return resp_mat, support_mat, lkh



        # load data
        def load_data(self,data_id):
            if data_id ==0:
                LSA_path = cmr_version_dir + 'datafile/autolab_GloVe.txt'
                data_path = cmr_version_dir + 'datafile/autolab_pres.txt'
                data_rec_path = cmr_version_dir + 'datafile/autolab_recs.txt'
                #data_cat_path = 'datafile/autolab_pres_cats.txt'
                subjects_path = cmr_version_dir + 'datafile/autolab_subject_id.txt'
            elif data_id==1:
                LSA_path = cmr_version_dir + 'datafile/K02_LSA.txt'
                data_path = cmr_version_dir + 'datafile/K02_data.txt'
                data_rec_path = cmr_version_dir + 'datafile/K02_recs.txt'
                subjects_path = cmr_version_dir + 'datafile/K02_list_ids.txt' # assume each list is a subject

            LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)        
            data_cat_path = cmr_version_dir + 'datafile/autolab_pres_cats.txt'
            return LSA_mat, data_path, data_rec_path, data_cat_path, subjects_path

        # data recoding 
        def data_recode(self, data_pres,data_rec):
            # recode data into lists without 0-paddings; convert to python indexing from matlab indexing by -1
            presents = [[int(y)-1 for y in list(x) if y>0] for x in list(data_pres)]
            recalls = [[int(y)-1 for y in list(x) if y>0] for x in list(data_rec)]

            # recode recall data into serial positions (set as x+100 if it is a repetition; set as -99 if it is an intrusion)
            recalls_sp = []
            for i,recall in enumerate(recalls):
                recall_sp = []
                this_list = presents[i]
                recallornot = np.zeros(len(this_list))
                for j,item in enumerate(recall):
                    try:
                        sp = this_list.index(item)
                        if recallornot[sp]==0: # non-repetition
                            recall_sp.append(this_list.index(item))
                            recallornot[sp]=1
                        else: # repetition
                            recall_sp.append(this_list.index(item)+100)
                    except:   # intrusion 
                        recall_sp.append(-99)

                recalls_sp.append(recall_sp)    
            return presents,recalls,recalls_sp


        # get stopping probability at each output position 
        def get_stopping_prob(self, recalls,list_length):
            lengths = [len(x) for x in recalls]
            counts = np.bincount(lengths)
            probs = []
            max_recall = np.max(lengths)
            for i in range(list_length):
                if i >= max_recall-1:
                    prob = 1
                else:    
                    prob = (counts[i]+1)/(counts[i]+np.sum(counts[i+1:])+1)
                probs.append(prob)

            return probs


        # spc and pfr
        def get_spc_pfr(self, recalls_sp,list_length):
            # this function returns serial position curve (spc) and prob of first recall (pfr) 
            # recalls_sp: list of lists of serial positions    
            num_trial = len(recalls_sp)
            spc = np.zeros(list_length)
            pfr = np.zeros(list_length)
            for i,recall_sp in enumerate(recalls_sp):
                recallornot = np.zeros(list_length)
                for j,item in enumerate(recall_sp):           
                    if 0 <= item <100:
                        #print(item)
                        if recallornot[item]==0:
                            spc[item] += 1
                            if j==0:
                                pfr[item] += 1 
                            recallornot[item]=1    

            return spc/num_trial, pfr/num_trial

        # crp
        def get_crp(self, recalls_sp,lag_examine,ll):
            # this function returns conditional response probability 
            # recalls_sp: list of lists of serial positions
            # lag_examine: range of lags to examine, can it to 4 usually
            # ll: list length
            possible_counts = np.zeros(2*lag_examine+1)
            actual_counts = np.zeros(2*lag_examine+1)
            for i in range(len(recalls_sp)):
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
                            for k,item in enumerate(recallornot):    
                                if item==0:
                                    lag = k - sp1
                                    if np.abs(lag) <= lag_examine:
                                        possible_counts[lag+lag_examine] += 1                   
            crp = [(actual_counts[i]+1)/(possible_counts[i]+1) for i in range(len(actual_counts))]
            crp[lag_examine] = 0
            return crp

        # get semantic associations for different lags
        def get_semantic(self, data_pres,data_rec,lag_number,LSA_mat):
            recalls = [[int(y)-1 for y in list(x) if y>0] for x in list(data_rec)]
            LSAs = []
            for l in lag_number:
                LSA = [0]
                for i in range(len(recalls)):
                    if len(recalls[i]) > l:
                        for j in range(len(recalls[i]))[:-l]:
                            sim = LSA_mat[recalls[i][j],recalls[i][j+l]]
                            LSA.append(sim)   
                LSAs.append(np.mean(LSA))     
            return LSAs

        def normed_RMSE(self, A,B): 
        # return the normed root mean square error
        # normed by the scale of A (the target dataset)
            mse = np.mean(np.power(np.subtract(A,B),2))
            normed_rmse = np.sqrt(mse)/(np.max(A)-np.min(A))
            return normed_rmse

        def normed_RMSE_singlevalue(self, A,B): 
        # return the normed root mean square error
        # normed by the value of A (the target dataset)
            mse = np.mean(np.power(np.subtract(A,B),2))
            normed_rmse = np.sqrt(mse)#/np.abs(A)
            return normed_rmse



        def plot_results(self, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs):
                ###############
            #
            #   Plot results
            #
            ###############
            plt.rcParams['figure.figsize'] = (20,5)

            SMALL_SIZE = 14
            MEDIUM_SIZE = 14
            BIGGER_SIZE = 14
            plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
            colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']

            plt.subplot(1,4,1)
            plt.plot(data_spc,'k')
            data=np.asarray(CMR_spcs)
            y = np.mean(data,0)
            error = np.std(data,0)
            plt.plot(range(len(data_spc)), y, 'k', color='#CC4F1B')
            plt.legend(['Data','CMR'])
            plt.fill_between(range(len(data_spc)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            plt.title('SPC')

            plt.subplot(1,4,2)
            plt.plot(data_pfr,'k')
            data=np.asarray(CMR_pfrs)
            y = np.mean(data,0)
            error = np.std(data,0)
            plt.plot(range(len(data_pfr)), y, 'k', color='#CC4F1B')
            plt.fill_between(range(len(data_pfr)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            plt.title('PFR')

            plt.subplot(1,4,3)
            plt.plot(data_crp,'*',color='k')
            data=np.asarray(CMR_crps)
            y = np.mean(data,0)
            error = np.std(data,0)
            plt.plot(range(len(data_crp)), y, 'k', color='#CC4F1B')
            plt.fill_between(range(len(data_crp)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            plt.title('CRP')

            plt.subplot(1,4,4)
            plt.plot(data_LSA,'*',color='k')
            data=np.asarray(CMR_LSAs)
            y = np.mean(data,0)
            error = np.std(data,0)
            plt.plot(range(len(data_LSA)), y, 'k', color='#CC4F1B')
            plt.fill_between(range(len(data_LSA)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            plt.title('Semantic clustering')

        def simulateCMR(self, resps, N, ll, lag_examine,LSA_mat, data_path, data_rec_path, subjects_path):    
            ###############
            #
            #   Simulate probCMR for N times to pool behavioral data before plotting 
            #
            ###############  


            RMSEs = []
            CMR_spcs = []
            CMR_pfrs = []
            CMR_crps = []
            CMR_intrusions = []
            CMR_crp_sames = []
            CMR_crp_diffs = []
            CMR_LSAs = []
            CMR_sprobs = []

            data_pres = np.loadtxt(data_path, delimiter=',')
            data_rec = np.loadtxt(data_rec_path, delimiter=',')
            _,data_recalls,data_sp = self.data_recode(data_pres, data_rec)
            data_spc,data_pfr = self.get_spc_pfr(data_sp,ll)

            for itr in range(N):
                # run CMR2 on the data
                resp = resps[itr]

                # recode simulations  
                _,CMR_recalls,CMR_sp = self.data_recode(data_pres, resp)
                CMR_spc,CMR_pfr = self.get_spc_pfr(CMR_sp,ll)

                ###############
                #
                #   Calculate fit and behavioral data
                #
                ###############  
                RMSE = 0    
                # SPC
                RMSE += self.normed_RMSE(data_spc, CMR_spc)
                CMR_spcs.append(CMR_spc)

                # PFR
                RMSE += self.normed_RMSE(data_pfr, CMR_pfr)
                CMR_pfrs.append(CMR_pfr)

                # CRP
                data_crp = self.get_crp(data_sp,lag_examine,ll)
                CMR_crp = self.get_crp(CMR_sp,lag_examine,ll)
                RMSE += self.normed_RMSE(data_crp, CMR_crp) 
                CMR_crps.append(CMR_crp)

                # Intrusion       
                data_allrecalls = list(itertools.chain.from_iterable(data_sp))
                data_intrusion = data_allrecalls.count(-99)/(len(data_allrecalls)+1)   
                CMR_allrecalls = list(itertools.chain.from_iterable(CMR_sp))
                CMR_intrusion = CMR_allrecalls.count(-99)/(len(CMR_allrecalls)+1) 
                RMSE += self.normed_RMSE_singlevalue(data_intrusion, CMR_intrusion)    
                CMR_intrusions.append(CMR_intrusion)

               # semantic clustering
                data_LSA = self.get_semantic(data_pres,data_rec,[1,2,3,4],LSA_mat)
                CMR_LSA = self.get_semantic(data_pres,resp,[1,2,3,4],LSA_mat)
                RMSE += self.normed_RMSE(data_LSA, CMR_LSA) 
                CMR_LSAs.append(CMR_LSA)  

                # stopping probability
                data_sprob = self.get_stopping_prob(data_recalls,ll)
                CMR_sprob = self.get_stopping_prob(CMR_recalls,ll)
                RMSE += self.normed_RMSE(data_sprob, CMR_sprob) 
                CMR_sprobs.append(CMR_sprob)  

                RMSEs.append(RMSE) 
            return RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs



        def model_probCMR(self, N, ll, lag_examine,data_id):  
            """Error function that we want to minimize"""
            ###############
            #
            #   simulate free recall data
            #
            # N: 0 - obtain lists of recall, in serial positions
            # N: 1 - obtain likelihood given data
            # N >1 - plot behavioral data with error bar with N being the number of times in simulations
            #
            # ll: list length (ll=16)
            #
            # lag_examine: lag used in plotting CRP
            #
            ###############
            LSA_mat, data_path, data_rec_path, data_cat_path, subjects_path = self.load_data(data_id)
            data_pres = np.loadtxt(data_path, delimiter=',')
            data_rec = np.loadtxt(data_rec_path, delimiter=',')

            # current set up is for fitting the non-emot version of the model
            # model parameters
            if data_id==0:
                param_dict = {

                    'beta_enc':  0.3187893806764954,           # rate of context drift during encoding
                    'beta_rec':  0.9371120781560975,           # rate of context drift during recall
                    'beta_rec_post': 1,      # rate of context drift between lists
                                                    # (i.e., post-recall)

                    'gamma_fc': 0.1762454837715133,  # learning rate, feature-to-context
                    'gamma_cf': 0.5641689110824742,  # learning rate, context-to-feature
                    'scale_fc': 1 - 0.1762454837715133,
                    'scale_cf': 1 - 0.5641689110824742,


                    's_cf': 0.8834467032413329,       # scales influence of semantic similarity
                                            # on M_CF matrix

                    's_fc': 0.0,            # scales influence of semantic similarity
                                            # on M_FC matrix.
                                            # s_fc is first implemented in
                                            # Healey et al. 2016;
                                            # set to 0.0 for prior papers.

                    'phi_s': 2.255110764387116,      # primacy parameter
                    'phi_d': 0.4882977227079478,      # primacy parameter


                    'epsilon_s': 0.0,      # baseline activiation for stopping probability 
                    'epsilon_d': 2.2858636787518285,        # scale parameter for stopping probability 

                    'k':  6.744153399759922,        # scale parameter in luce choice rule during recall

                    # parameters specific to optimal CMR:
                    'primacy': 0.0,
                    'enc_rate': 1.0,

                }
            elif data_id==1:
                param_dict = {

                    'beta_enc':  0.7887626184661226,           # rate of context drift during encoding
                    'beta_rec':  0.49104864172027485,           # rate of context drift during recall
                    'beta_rec_post': 1,      # rate of context drift between lists
                                                    # (i.e., post-recall)

                    'gamma_fc': 0.4024001271645564,  # learning rate, feature-to-context
                    'gamma_cf': 1,  # learning rate, context-to-feature
                    'scale_fc': 1 - 0.4024001271645564,
                    'scale_cf': 0,


                    's_cf': 0.8834467032413329,       # scales influence of semantic similarity
                                            # on M_CF matrix

                    's_fc': 0.0,            # scales influence of semantic similarity
                                            # on M_FC matrix.
                                            # s_fc is first implemented in
                                            # Healey et al. 2016;
                                            # set to 0.0 for prior papers.

                    'phi_s': 4.661547054594787,      # primacy parameter
                    'phi_d': 2.738934338758688,      # primacy parameter


                    'epsilon_s': 0.0,      # baseline activiation for stopping probability 
                    'epsilon_d': 2.723826426356652,        # scale parameter for stopping probability 

                    'k':  5.380182482069175,        # scale parameter in luce choice rule during recall

                    # parameters specific to optimal CMR:
                    'primacy': 0.0,
                    'enc_rate': 1.0,

                }


            # run probCMR on the data
            if N==0: # recall_mode = 0 to simulate based on parameters
                resp, times,_ = self.run_CMR2(
                    recall_mode=0,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
                    params=param_dict, subj_id_path=subjects_path, sep_files=False)   
                _,CMR_recalls,CMR_sp = self.data_recode(data_pres, resp)
                return CMR_sp
            if N==1:# recall_mode = 1 to calculate likelihood of data based on parameters
                _, _,lkh = self.run_CMR2(
                    recall_mode=1,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
                    params=param_dict, subj_id_path=subjects_path, sep_files=False)   
                return lkh   
            else:
            
                resps = []
                for k in range(N):
                    resp, times,_ = self.run_CMR2(
                        recall_mode=0,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
                        params=param_dict, subj_id_path=subjects_path, sep_files=False)   
                    _,CMR_recalls,CMR_sp = self.data_recode(data_pres, resp)
                    resps.append(resp)

                RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs = self.simulateCMR(resps, N, ll, lag_examine, LSA_mat, data_path, data_rec_path, subjects_path)
                self.plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs)
                return -RMSE
            
        def separate_files(self, data_path,rec_path,subj_id_path):
            """If data is in one big file, separate out the data into sheets, by subject.

            :param data_path: If using this method, data_path should refer directly
                to a single data file containing the consolidated data across all
                subjects.
            :param subj_id_path: path to a list of which subject each list is from.
                lists from a specific subject should all be contiguous and in the
                order in which the lists were presented.
            :return: a list of data matrices, separated out by individual subjects.

            """

            # will contain stimulus matrices presented to each subject/recalled data for each subject
            subj_presented_data = []
            subj_recalled_data = []

            # for test subject
            data_pres_list_nos = np.loadtxt(data_path, delimiter=',')
            data_recs_list_nos = np.loadtxt(rec_path, delimiter=',')

            # get list of unique subject IDs

            # use this if dividing a multiple-session subject into sessions
            subj_id_map = np.loadtxt(subj_id_path)
            unique_subj_ids = np.unique(subj_id_map)

            # Get locations where each Subj's data starts & stops.
            new_subj_locs = np.unique(
                np.searchsorted(subj_id_map, subj_id_map))

            # Separate data into sets of lists presented to each subject
            for i in range(new_subj_locs.shape[0]):

                # for all but the last list, get the lists that were presented
                # between the first time that subj ID occurs and when the next ID occurs
                if i < new_subj_locs.shape[0] - 1:
                    start_lists = new_subj_locs[i]
                    end_lists = new_subj_locs[i + 1]

                # once you have reached the last subj, get all the lists from where
                # that ID first occurs until the final list in the dataset
                else:
                    start_lists = new_subj_locs[i]
                    end_lists = data_pres_list_nos.shape[0]

                # append subject's sheet
                subj_presented_data.append(data_pres_list_nos[start_lists:end_lists, :])
                subj_recalled_data.append(data_recs_list_nos[start_lists:end_lists, :])

            return subj_presented_data, subj_recalled_data, unique_subj_ids
        
    return Functions