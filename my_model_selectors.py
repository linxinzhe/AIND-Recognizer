import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        min_score = float("inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)

                logL = model.score(self.X, self.lengths)

                # number of params = initial probabilities + transition probabilities + emission probabilities
                # initial = n-1
                # transition = n*(n-1)
                # emission = n*d+n*d  which d means features in each observation
                p = -1 + n ** 2 + n * len(self.X[0]) * 2

                logN = math.log(len(self.X))

                BIC = -2 * logL + p * logN

                if BIC < min_score:
                    min_score = BIC
                    best_model = model
            except:
                pass
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        max_score = float("-inf")
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                sum_anti_logL = 0
                word_count = 0

                model = self.base_model(n)
                if model:
                    logL = model.score(self.X, self.lengths)

                    for word in self.hwords:
                        if word != self.this_word:
                            sum_anti_logL += model.score(*self.hwords[word])
                            word_count += 1

                    mean_anti_logL = sum_anti_logL / word_count

                    DIC = logL - mean_anti_logL

                    if DIC > max_score:
                        max_score = DIC
                        best_model = model
            except:
                pass
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        max_score = float("-inf")
        best_model = None

        split_method = KFold(min(len(self.sequences), 3))  # api default = 3
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)

                logLs = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                        test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                        model.fit(train_X, train_lengths)
                        logL = model.score(test_X, test_lengths)
                        logLs.append(logL)

                    except:
                        pass
                mean_score = np.mean(logLs)

                if mean_score > max_score:
                    max_score = mean_score
                    best_model = model
            except:
                pass

        return best_model
