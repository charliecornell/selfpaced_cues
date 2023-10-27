import numpy as np
import scipy.io
import math
from glob import glob
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import CMRversions.probCMR as probCMR


class CMR2Reminder(probCMR.CMR2):
        
    
    
    def get_context_space(self,postcue=False):
        """
        Obtain encoding context vector for every item from the list.
        Note this is not apart of the model functioning, but only later returned for analysis.
        """
        net_cs = []
        if postcue: thislist_pattern = self.pres_list_nos[self.list_idx-1]
        else: thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(self.all_session_items_sorted, thislist_pattern)
        
        for i in range(self.listlength):
            idx = thislist_pres_indices[i]
            f_net_temp = np.zeros([1,self.nelements], dtype=np.float32)
            f_net_temp[0][idx] = 1
            net_c = np.dot(self.M_FC_tem*self.params['gamma_fc']+self.M_FC_sem*(1-self.params['gamma_fc']), f_net_temp.T)
            nelements_temp = self.nstudy_items_presented + 2*self.nlists + 1
            cin_temp = net_c[:nelements_temp]
            c_normed = probCMR.norm_vec(cin_temp)
            net_cs.append(c_normed)
            
        return net_cs
    
            
    
    def recall_session(self):
        """Simulate a recall portion of an experiment, following a list presentation. """
        
        self.beta_in_play = self.params['beta_rec']  
        nitems_in_session = self.listlength * self.nlists # potential items to recall

        # initialize list to store recalled items
        recalled_items = []
        support_values = []
        
        # track what are the items to recall for this particular list, including potential extralist items
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(self.all_session_items_sorted, thislist_pattern)
        self.torecall = np.zeros([1, nitems_in_session], dtype=np.float32)
        self.torecall[0][thislist_pres_indices] = 1
          
        if self.recall_mode == 0: # simulations
            continue_recall = 1
            
            # start the recall with cue position if not negative
            if self.params['cue_position'] >= 0: 
                self.c_net = self.c_cue.copy()
       
            # simulate a recall session/list 
            while continue_recall == 1:

                # get item activations to input to the accumulator
                f_in = self.obtain_f_in() # obtain support for items

                # recall process:
                winner_idx, support = self.retrieve_next(f_in.T)
                winner_ID = np.sort(self.all_session_items)[winner_idx]

                # if item retrieved, recover item info corresponding to activation value index retrieved by accumulator
                if winner_idx is not None:
                    recalled_items.append(winner_ID)
                    support_values.append(support)
                    self.present_item(winner_idx)     # reinstantiate this item
                    self.torecall[0][winner_idx] = -1 # update the to-be-recalled remaining items
                    self.update_context_recall()      # and update context
                else:
                    continue_recall = 0
                
        else: # force recall order and calculate probability of a known recall list
            thislist_recs = self.recs_list_nos[self.list_idx]
            thislist_recs_indices = np.searchsorted(self.all_session_items_sorted, thislist_recs)
            recall_length = np.count_nonzero(thislist_recs)
            
            for i in range(recall_length):
                recall_idx = thislist_recs_indices[i]
                recall_ID = np.sort(self.all_session_items)[recall_idx]
                recalled_items.append(recall_ID) # recalled items, sorted by wordpool index
                
                if recall_idx < len(self.all_session_items_sorted):
                    f_in = self.obtain_f_in() 
                    self.lkh += self.retrieve_probability(f_in.T,recall_idx,0) # recall process
                    self.present_item(recall_idx)     # reinstantiate this item
                    self.torecall[0][recall_idx] = -1 # update the to-be-recalled remaining items
                    self.update_context_recall()      # and update context
                else:
                    self.count += 1

            f_in = self.obtain_f_in() 
            self.lkh += self.retrieve_probability(f_in.T,0,1) # stopping process
            
        # MODIFIED: update counter of what list we're on; commmented out because now at end of reminder session
        # self.list_idx += 1
        
        return recalled_items, support_values
    
    
    
    def reminder_session(self):
        """Simulate a cued recall portion of an experiment, following a list presentation and any prior recalls."""
        
        self.beta_in_play = self.params['beta_rec']  
        nitems_in_session = self.listlength * self.nlists # potential items to recall

        # track what are the items to recall for this particular list, including potential extralist items
        thislist_pattern = self.pres_list_nos[self.list_idx] # presentation order given by wordpool values
        thislist_pres_indices = np.searchsorted(self.all_session_items_sorted, thislist_pattern) # indices if pres was sorted
        thislist = thislist_pattern.tolist()
        recalled = self.torecall.copy()
        reminders = [i for i in range(nitems_in_session) if recalled[0][i]==1 ]
        
        recalls_sp = []
        reminders_sp = []
        accs = []
        
        for i in range(len(reminders)): 
            acc = []
            reminder = reminders[i]
            reminder_ID = np.sort(self.all_session_items)[reminder]
            reminders_sp.append(thislist.index(reminder_ID))
            
            reps = 35 # repetitions for each reminder
            for r in range(reps):
                
                self.torecall = recalled.copy()              # reset which words are remaining
                self.present_item(reminder)                  # present external cue
                self.torecall[0][reminder] = -1              # mark as recalled so not to be retrieved
                self.beta_in_play = 1                        # drift rate to fully set context to cue
                self.update_context_recall()                 # update context
                self.beta_in_play = self.params['beta_rec']  # set back to regular beta for recall
                recalled_items = []                          # initialize list to store recalled items

                if self.recall_mode<2: # simulations 
                    continue_recall = 1
                    
                    # simulate a recall session/list 
                    while continue_recall == 1:

                        # get item activations to input to the accumulator
                        f_in = self.obtain_f_in() # obtain support for items

                        # recall process:
                        winner_idx, support = self.retrieve_next(f_in.T)
                        winner_ID = np.sort(self.all_session_items)[winner_idx]

                        # if item retrieved, recover item info corresponding to activation value index retrieved by accumulator
                        if winner_idx is not None:
                            recalled_items.append(thislist.index(winner_ID))
                            self.present_item(winner_idx)     # reinstantiate this item
                            self.torecall[0][winner_idx] = -1 # update the to-be-recalled remaining items
                            self.update_context_recall()      # and update context
                        else:
                            continue_recall = 0
                                
                acc.append(len(recalled_items))
                
            accs.append(acc)
            recalls_sp.append(recalled_items) # only store the last repetition of reminder session recall

        context_tem = self.M_CF_tem[:,thislist_pres_indices]
        context_sem = self.M_CF_sem[:,thislist_pres_indices]
        context_all = context_tem + context_sem
        pca = PCA(n_components=2)
        contexts_tem = pca.fit_transform(context_tem.T)
        contexts_sem = pca.fit_transform(context_sem.T)
        contexts_pca = np.concatenate((contexts_tem, contexts_sem), axis=1) # separate temporal and semantic contexts
        contexts_pca2= pca.fit_transform(context_all.T) # encoding contexts
        
        # update counter of what list we're on
        self.list_idx += 1

        return reminders_sp, recalls_sp, accs, contexts_pca, contexts_pca2
        
        
        