"""
The class SMOTE is a Python implementation of SMOTE.
This implementation is based on the original variant of SMOTE.
Original paper: https://arxiv.org/pdf/1106.1813.pdf

The functions get_uncertainty_samples() and get_diversity_samples are based on
the chapter 3 "Uncertainty Sampling" and 4 "Diversity Sampling" of the book
Human-in-the-Loop Machine Learning by Rob Munro
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

class SMOTE:
    """Python implementation of SMOTE.
        This implementation is based on the original variant of SMOTE.
        Parameters
        ----------
        ratio : int, optional (default=100)
            The ratio percentage of generated samples to original samples.
            - If ratio < 100, then randomly choose ratio% of samples to SMOTE.
            - If ratio >= 100, it must be a interger multiple of 100.
        k_neighbors : int, optional (defalut=6)
            Number of nearest neighbors to use to SMOTE.
        random_state : int, optional (default=None)
            The random seed of the random number generator.
    """
    def __init__(self,
                 ratio=100,
                 k_neighbors=6,
                 random_state=None):

        # Check input arguments
        if ratio > 0 and ratio < 100:
            self.ratio = ratio
        elif ratio >= 100:
            if ratio % 100 == 0:
                self.ratio = ratio
            else:
                raise ValueError(
                    'ratio over 100 should be multiples of 100')
        else:
            raise ValueError(
                'ratio should be greater than 0')

        if type(k_neighbors) == int:
            if k_neighbors > 0:
                self.k_neighbors = k_neighbors
            else:
                raise ValueError(
                    'k_neighbors should be integer greater than 0')
        else:
            raise TypeError(
                'Expect integer for k_neighbors')

        # Set random seed so results are reproducible
        if type(random_state) == int:
            np.random.seed(random_state)

    
    def _sample_randomly(self, samples, ratio):
        """
        Sample randomly a ndarray 
            ratio >= 100 randomly oversample the ndarray samples (with replacement)
            ratio < 100 randomly undersample the ndarray samples (with replacement)
        ----------
        samples : list or ndarray, shape (n_samples, n_features)
            The samples to apply SMOTE to.
        ratio : int, optional (default=100)
            The ratio percentage of generated samples to original samples.
            - If ratio < 100, then randomly choose ratio% of samples to SMOTE.
            - If ratio >= 100, it must be a interger multiple of 100.
        Returns
        -------
        output : ndarray of the sampling done to the ndarray samples
        """
        print("I entered the function _sample_randomly")
        length = samples.shape[0]
        target_size = int(length * ratio)
        idx = np.random.randint(length, size=target_size)

        return samples[idx, :]

    def _generate_syntethic_samples(self, idx, nnarray):
        """
        Generate Synthetic Samples SMOTE
        Parameters
        ----------
        idx : int, index number
        nnarray : ndarray, k-nearest of the samples
        Returns
        -------
        output : ndarray
            Smote samples generated from the K_neigbor of samples.
        """
        for i in range(self.N):
            nn = np.random.randint(low=0, high=self.k_neighbors) # Chose a random K-Neighbor
            for attr in range(self.numattrs): 
                dif = (self.samples_all_space[nnarray[nn]][attr] - self.samples[idx][attr])
                gap = np.random.uniform()
                self.smote_samples[self.newidx][attr] = (self.samples[idx][attr] + gap * dif)                                                
            self.newidx += 1

        return self.smote_samples

            

    def oversample(self, samples, samples_all_space, merge=False):
        """Perform oversampling using SMOTE
        Parameters
        ----------
        samples : list, ndarray or Pandas DataFrame, shape (n_samples, n_features)
            The samples to apply SMOTE to.
        merge : bool, optional (default=False)
            If set to true, merge the smote_samples to original samples.
        Returns
        -------
        output : ndarray
            The new generated smote_samples.
        """
        # Validate type of samples (ndarray is expected)
        if type(samples) == list:
            self.samples = np.array(samples)
        elif type(samples) == pd.DataFrame:
            self.samples = samples.to_numpy()
        elif type(samples) == np.ndarray:
            self.samples = samples
        else:
            raise TypeError(
                'Expect a built-in list, a pandas DataFrame or an ndarray for samples')

        if type(samples_all_space) == list:
            self.samples_all_space = np.array(samples_all_space)
        elif type(samples_all_space) == pd.DataFrame:
            self.samples_all_space = samples_all_space.to_numpy()
        elif type(samples_all_space) == np.ndarray:
            self.samples_all_space = samples_all_space
        else:
            raise TypeError(
                'Expect a built-in list, a pandas DataFrame or an ndarray for samples')

        self.numattrs = self.samples.shape[1] # Set the num of features of the ndarray

        # Undersample the samples based on ratio
        if self.ratio < 100:
            ratio = ratio / 100.0
            self.samples = self._sample_randomly(self.samples, ratio) 
            self.ratio = 100

        self.N = int(self.ratio / 100) # N is how many more SMOTE samples are going to be generated 
        new_shape = (self.samples.shape[0] * self.N, self.samples.shape[1])
        self.smote_samples = np.empty(shape=new_shape)
        self.newidx = 0

        self.nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nbrs.fit(self.samples_all_space)
        distances, self.knn = self.nbrs.kneighbors(self.samples) # Return indices of the Nearest Neighbors

        for idx in range(self.samples.shape[0]):
            nnarray = self.knn[idx]
            self._generate_syntethic_samples(idx, nnarray)

        if merge:
            return np.concatenate((self.samples, self.smote_samples))
        else:
            return self.smote_samples    

def get_uncertainty_samples(model, X, uncertainty_measure, perc_uncertain=0.5, col_minority_class=0):
    """ 
    Obtain the % of X most uncertain for the model
    
    Parameters
    ----------
      model : sklearn model, previously train with the distribution of X.
      X : Pandas DataFrame, where the confidence of the model is going to be calculated.
      uncertainty_measure : string, stating the uncertainty sampling technique to use "least_confidence", "margin_confidence", "entropy" supported
      perc_uncertain : float, that is the % of X that is going to be retained (most unconfident samples)  
      col_minority_class : int, of the position of the minority class
    
    Returns
    -------
    output : Pandas DataFrame
      % of X of most unconfident samples
    """
    # Check input arguments
    if type(X) == pd.DataFrame:
      pass
    else: 
      raise TypeError('Expect pandas DataFrame but: ' + str(type(X)) + " was given")

    if "sklearn" not in str(type(model)):
      raise TypeError("An sklearn trained model was expected")
    else:
      pass

    if type(uncertainty_measure) == str:
      pass
    else:
      print("The uncertainty measure has to be a string")

    if (perc_uncertain > 1) | (perc_uncertain < 0):
      print("The perc_uncertain has to be a number between 0 and 1")
    else:
      pass

    # Caculate probabilities with the model
    prob_dist = model.predict_proba(X) 
    prob_dist_minority_class = prob_dist[:, col_minority_class] # Probability of belonging to minority class (0 index may change in other datasets)
    num_labels = prob_dist.shape[1] 

    # Calculate uncertainty score base on model probabilities
    if uncertainty_measure == "least_confidence":
      simple_least_conf = 1 - prob_dist_minority_class
      normalized_least_conf1 = simple_least_conf * (num_labels / (num_labels - 1))
      uncertainty_score = normalized_least_conf1
    elif uncertainty_measure == "margin_confidence":
      difference = abs(prob_dist[:, 0] - prob_dist[:, 1])
      uncertainty_score = 1 - difference
    elif uncertainty_measure == "entropy":
      log_probs = prob_dist * np.log2(prob_dist)
      raw_entropy = 0 - np.sum(log_probs, axis=1)
      normalized_entropy = raw_entropy / np.log2(num_labels)
      uncertainty_score = normalized_entropy
    else:
      print("The uncertainty measure " + uncertainty_measure + " is not supported")

    # Keep the indexes of X for reproducibility
    df_confidence = pd.concat([pd.DataFrame(X.index), pd.DataFrame(uncertainty_score, columns=[uncertainty_measure])], axis=1)
    df_confidence.set_index('index', inplace=True)

    # Sort and keep the most unconfident samples
    df_confidence_sorted = df_confidence.sort_values(uncertainty_measure, ascending=False)
    number_rows = int(len(df_confidence_sorted)*perc_uncertain) # First n rows most uncertain
    df_confidence_sorted = df_confidence_sorted.head(number_rows)

    # Obtain training data where the model is most uncertain
    X_uncertainty_sample = X[X.index.isin(df_confidence_sorted.index)] 

    return X_uncertainty_sample

def get_diversity_samples(X, perc_sample=0.1, n_clusters=5, random_state=1):
    """ 
    Creates kmeans clusters from X and then sample uniformly from those clusters. This creates a diversity sampling 
    
    Parameters
    ----------
      X : Pandas DataFrame, to cluster and to sample .
      perc_sample: float, % of how much of X is going to be sample.
      n_clusters: int, kmeans clusters to be generated.
      random_state: int, random seed to produce reproducible results.
    Returns
    -------
    output : Pandas DataFrame
      Uniform sample of X based on n_clusters of kmeans
    """
    # Check input arguments
    if type(X) == pd.DataFrame:
      pass
    else: 
      raise TypeError('Expect pandas DataFrame but: ' + str(type(X)) + " was given")

    if (perc_sample > 1) | (perc_sample < 0):
      print("The perc_sample has to be a number between 0 and 1")
    else:
      pass

    if type(n_clusters) == int:
      pass
    else:
      print("n_clusters has to be an int")

    if type(random_state) == int:
      pass
    else:
      print("random_state has to be an int")


    # Create clusters with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)

    # Keep Indexes of X for reproducibility
    df_kmeans = pd.concat([pd.DataFrame(X.index), pd.DataFrame(kmeans.labels_, columns=["kmeans_cluster"])], axis=1)
    df_kmeans.set_index('index', inplace=True)

    # Sample uniformly from each cluster (This creates a diversity sampling)
    number_samples_per_cluster = round(int(len(df_kmeans)*perc_sample)/5) 
    try:
        kmeans_samples = df_kmeans.groupby('kmeans_cluster', group_keys=False).apply(lambda x: x.sample(number_samples_per_cluster, random_state=random_state))
    except ValueError:
        min_samples_cluster = df_kmeans.value_counts().min()
        num_clusters = df_kmeans.value_counts().shape[0]
        perc_to_sample = round((num_clusters*min_samples_cluster) / df_kmeans.shape[0] *100, 2)
        print("There is a cluster with only " + str(min_samples_cluster) + " samples, to have a significant representation of every cluster we can only sample maximum " + str(perc_to_sample)+ "% of the data.")
        print("Setting the number of samples per cluster to be: " + str(min_samples_cluster))
        kmeans_samples = df_kmeans.groupby('kmeans_cluster', group_keys=False).apply(lambda x: x.sample(min_samples_cluster, random_state=random_state))
        
    # Obtain training data where the model is most uncertain
    X_diversity_sample = X[X.index.isin(kmeans_samples.index)] 

    return X_diversity_sample