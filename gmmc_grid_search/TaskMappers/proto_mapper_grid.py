import numpy as np
from argparse import Namespace
import sys

import TaskMappers.gmmc as gmmc
sys.path.append("../")
from utils import log


class ProtoTaskMapper():
    def __init__(self, mapper_params, **llmkwargs):
        
        '''
        ** Optionally, set n_C_max to number of classes within Task
        Can be GMM or NM + PCA (optional)
        Can be done on full train data or data subset 
        '''
        self.param = Namespace(**mapper_params)
        
        self.n_C_min = self.param.n_cluster_min
        self.n_C_max = self.param.n_cluster_max
        
        self.proto2task_memberships={} # mapping of proto ID to Task ID (many to one)
        
        self.numProto=[0]

        if self.param.task_mapper_type=='gmm':
            self.ProtoMap = gmmc.LLGMM(self.param.covariance_type, **llmkwargs)

            
    def eval_model(self, bics, n_centroids):
        ''' 
        Get best estimate of number of centroids using BIC
        Can adapt to be some other measure such as silhouette 
        '''
        grads = np.gradient(np.gradient(bics))
        chosen_Model = np.argsort(grads)[::-1][0]
        chosen_K = n_centroids[chosen_Model]
        return chosen_Model, chosen_K

    def fit_task(self, task_num, n_c, taskID, task_features, **llmkwargs):
        '''
        Generates new gmm models and evaluates which is appropriate (K) according to BIC 
        For gmm, generate all gmm parameters including Cov and weights as well as means
        Returns the initilizing centroids 
        '''
        n_c_log = n_c
        # TODO can add PCA here per Task or Per-class on task_features 
        
        
        if self.n_C_min < self.n_C_max:
            # ---- generate excess est_centers_new (can be xmeans or gmms)
            temp_store_model = []
            bic_scores = []
            n_centroids = np.arange(self.n_C_min, self.n_C_max+1)
            # Use the Bic score metric to select number of centroids 
            for n_c in n_centroids:
                out_fit = self.ProtoMap.fit_task(task_features, n_c, **llmkwargs)
                temp_store_model.append(out_fit[:-1])
                bic_scores.append(out_fit[-1])
                
            gmm_ind, _ = self.eval_model(bic_scores, n_centroids)
            gmm_model_new = temp_store_model[gmm_ind]
        else:
            #n_c = self.n_C_min
            log(f"t{task_num}_clu{n_c_log}", f"this task has: {n_c}")
            gmm_ind=0
            prototypes, covariances, counts, bic_s = self.ProtoMap.fit_task(task_features, n_c, **llmkwargs)
            gmm_model_new = (prototypes, covariances, counts)
            
        log(f"t{task_num}_clu{n_c_log}", 'Generate %d New Centroids'%gmm_model_new[0].shape[0])
        
        # Integrate into mamoery
        self.ProtoMap.consolidate_fit(*gmm_model_new)
        
        for p in range(self.ProtoMap.nC[-2], self.ProtoMap.nC[-1]): ## task mapper 
            self.proto2task_memberships[p]=taskID
        log(f"t{task_num}_clu{n_c_log}", f"All Centroids so far: {self.ProtoMap.w.shape[0]} {self.ProtoMap.LLClustering.weights_} {sum(self.ProtoMap.LLClustering.weights_)}")    

    def predict_task(self, task_features_sample):
        
        
        
        index_proto = self.ProtoMap.predict_proto(task_features_sample)

        index_proto = int(index_proto)
        task_predicted = self.proto2task_memberships[index_proto]
        
        return task_predicted



    