import numpy as np

from sklearn.mixture import GaussianMixture


class LLGMM():
    def __init__(self, covariance_type, **kwargs):
        
        '''Does Incremental Task mapping by GMM clustering + Dict mapper'''


        self.nC = [0]
        self.w = None
        self.w_std = None ## store the inverse, for prediction. Cholensky cov 
        self.mixtures = None
        self.covariance_type = covariance_type
        
        self.LLClustering = GaussianMixture(n_components=1,covariance_type=self.covariance_type, **kwargs) # start with one cluster but modify straigh away
                        
    def append_prototypes(self, new_protypes, new_stds, new_counts):
        
        if self.w is not None and self.w_std is not None:
            self.w = np.vstack((self.w, new_protypes))
            self.w_std = np.vstack((self.w_std, new_stds))
            self.counts = np.concatenate((self.counts, new_counts))
        else:
            self.w = new_protypes
            self.w_std = new_stds
            self.counts = new_counts
                    
    def fit_task(self, features_pertask, n_cluster, **kwargs):
                
        GMM_clustering = GaussianMixture(n_components=n_cluster,covariance_type=self.covariance_type, **kwargs)
        
        GMM_clustering.fit(features_pertask)
                
        prototypes = GMM_clustering.means_ 
        covariances = GMM_clustering.precisions_cholesky_
        counts = (GMM_clustering.weights_)*features_pertask.shape[0]
        
        bic_scores = GMM_clustering.bic(features_pertask)
        
        return prototypes, covariances, counts, bic_scores
        
        
    def consolidate_fit(self, prototypes, covariances, counts):
        
        self.append_prototypes(prototypes, covariances, counts)
        
        # ============modify original GMM ==========
        self.LLClustering.means_ = self.w
        self.LLClustering.precisions_cholesky_ = self.w_std
        self.LLClustering.weights_ = self.counts/sum(self.counts)
        self.LLClustering.n_components = self.w.shape[0]
                
        self.nC.append(self.LLClustering.n_components)
        
        
    def predict_proto(self, sample):
        
        if len(sample.shape)==1:
            sample = np.expand_dims(sample,0)
        
        self.distances = self.LLClustering.predict_proba(sample)

        category = np.argmax(self.distances)

        return int(category)


