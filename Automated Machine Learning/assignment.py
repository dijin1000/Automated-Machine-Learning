# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:19:00 2019

@author: Gerar
"""

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.model_selection
import sklearn.tree

import typing


class SurrogatedGreedyDefaults(object):

    def __init__(self):
        # Both variables to be set by "train_surrogates"
        self.surrogates = dict()
        self.hyperparameter_columns = None

    def train_surrogates(self, meta_data: pd.DataFrame,
                         hyperparameter_columns: typing.List[str],
                         performance_column: str,
                         task_indication_column: str) -> typing.NoReturn:
        """
        Takes a pandas data frame containing meta data as input, where each row represents a configuration on a task,
        each column represents either a hyperparameter value, the performance measure or the task it was performed on.
        Stores for each task that it encounters a surrogate model in the dict `surrogates', mapping from task id to
        the surrogate model of that specific task. Each surrogate model is trained to map the inputs to the return
        value.

        :param meta_data: the dataframe with the meta-data
        :param hyperparameter_columns: list of column names that represent the surrogate
        :param performance_column: the column name that represents the performance of the configuration
        :param task_indication_column: the column name that represents the task the configuration was ran on
        """
        for i in meta_data[task_indication_column].unique():
            tmp_subset = meta_data.loc[meta_data[task_indication_column] == i]
            tmp_regr = sklearn.tree.DecisionTreeRegressor(random_state=1)
            tmp_regr.fit(tmp_subset[hyperparameter_columns], tmp_subset[performance_column])
            self.surrogates[i] = tmp_regr

    def surrogate_predict(self, task_id: int, configurations: np.array) -> np.array:
        """
        Uses the surrogate of a specific task to predict how well a certain configuration would work

        :param task_id: the task id
        :param configurations: numpy array with inputs for the surrogate model. The array will be of dimensions (N, D)
        :return: the prediction of the surrogate how the configurations will perform on this task. The array will be of
        dimension (N,)
        """
        return self.surrogates[task_id].predict(configurations)

    def determine_defaults(self, configurations: typing.List[typing.Dict[str, typing.Any]],
                           aggregate: typing.Optional[typing.Callable]) -> typing.List:
        """
        For a list of configurations, determines greedily which of these are a good set of defaults.

        :param configurations: the set of configurations
        :param aggregate: the aggregation function (will only be used for greedy defaults, ignored for average rank)
        :return: a list of configurations, that will work as defaults
        """        
        results = pd.DataFrame([])
        for conf in configurations:
            results = results.append(pd.DataFrame([[surr.predict([list(conf.values())]) for surr in self.surrogates.values()]]))        
        results = pd.DataFrame(configurations).join(results.reset_index(drop=True))      
        results = results.set_index(list(configurations[0].keys()))

        possibilities = results
        rankedRows = pd.DataFrame([])
        improve = True
        prevMax = 0
        while improve == True:
            tempDict = {}
            for index, row in possibilities.iterrows():                
                tmpData = ranked.append(row)
                tmpData = tmpData.apply(max)
                tmpData = tmpData.apply(aggregate,axis = 1)
                tempDict[index] = tempData.iloc[0]   
            indexer = max(tempDict, key=tempDict.get)
            value = tempDict[indexer]
            ranked = ranked.append(possibilities.loc[[indexer]])
            possibilities = possibilities.drop(ranked.index.tolist()[-1])  
            if(value <= prevMax):
                improve = False
            else:
                prevMax = value
        return(ranked.index.tolist())


    @staticmethod
    def evaluate(classifier: sklearn.base.BaseEstimator,
                 defaults: typing.List, X_train, X_test, y_train, y_test) -> float:
        """
        Takes the defaults as generated by the fit method, and evaluates them on a specific test task. The test task
        is a set of two numpy arrays, X being an N times D matrix (N being the number of observations, and D being the
        number of attributes), y being a N sized vector. Hint: have a look at the sklearn.model_selection.GridSearchCV
        classifier in scikit-learn (use it with default hyperparameters).

        :param classifier: the classifier that is being optimized (has a fit, predict and set_params method)
        :param defaults: the defaults, as generated by the fit method
        :param X_train: the data to train the classifier on
        :param X_test: the data to test the classifier on
        :param y_train: the train labels
        :param y_test: the test labels
        :return: the best obtained performance
        """
        search_cv = sklearn.model_selection.GridSearchCV(classifier, defaults)
        search_cv.fit(X_train, y_train)
        return search_cv.score(X_test, y_test)




