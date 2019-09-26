import ConfigSpace  # developed in Freiburg. installation: pip install Cython; pip install ConfigSpace
import pandas as pd
import sklearn.svm
import sklearn.datasets
import unittest

from assignment import SurrogatedGreedyDefaults


class TestMetaModels(unittest.TestCase):

    @staticmethod
    def sample_configurations(n_configurations, seed):
        # function uses the ConfigSpace package, as developed at Freiburg University.
        # most of this functionality can also be achieved by the scipy package
        # same hyperparameter configuration as in scikit-learn
        cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', seed)

        C = ConfigSpace.UniformFloatHyperparameter(
            name='C', lower=0.03125, upper=32768, log=True, default_value=1.0)
        gamma = ConfigSpace.UniformFloatHyperparameter(
            name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
        cs.add_hyperparameters([C, gamma])

        if n_configurations == 1:  # flaw in ConfigSpace library
            return [cs.sample_configuration(n_configurations).get_dictionary()]
        else:
            return [configuration.get_dictionary() for configuration in cs.sample_configuration(n_configurations)]

    def test_surrogate_prediction(self):
        # if this function does not succeed, please fix train_surrogates
        with open('data_svm_rbf.csv', 'r') as fp:
            df = pd.read_csv(fp)

        meta_model = SurrogatedGreedyDefaults()
        meta_model.train_surrogates(df, ['C', 'gamma'], 'predictive_accuracy', 'task_id')
        random_configurations = pd.DataFrame(TestMetaModels.sample_configurations(5, 42))

        prediction_6 = meta_model.surrogate_predict(6, random_configurations[['C', 'gamma']])
        prediction_11 = meta_model.surrogate_predict(11, random_configurations[['C', 'gamma']])
        prediction_12 = meta_model.surrogate_predict(12, random_configurations[['C', 'gamma']])

        fixture_6 = [0.8274, 0.91905, 0.19515, 0.9795125, 0.938]
        fixture_11 = [0.9168, 0.9168, 0.9968, 0.9536, 0.9088]
        fixture_12 = [0.1245, 0.455, 0.105, 0.1075, 0.106]

        for idx in range(5):
            self.assertAlmostEqual(prediction_6[idx], fixture_6[idx])
            self.assertAlmostEqual(prediction_11[idx], fixture_11[idx])
            self.assertAlmostEqual(prediction_12[idx], fixture_12[idx])

    def test_defaults(self):
        with open('data_svm_rbf.csv', 'r') as fp:
            df = pd.read_csv(fp)
        hyperparameters = ['C', 'gamma']
        meta_model = SurrogatedGreedyDefaults()
        meta_model.train_surrogates(df, hyperparameters, 'predictive_accuracy', 'task_id')
        random_configurations = TestMetaModels.sample_configurations(32, 42)

        defaults = meta_model.determine_defaults(random_configurations, sum)
        print(defaults)
        fixture = [
            (129.9939107327918, 0.00013990045035991287),
            (21591.81713748779, 2.5807085790756554),
            (0.041570095885706176, 4.686899482804055e-05),
            (5125.6630099979975, 0.007405151091678942)
        ]
        for n_default in range(len(fixture)):
            for n_param in range(len(fixture[0])):
                self.assertAlmostEqual(defaults[n_default][n_param], fixture[n_default][n_param])

        list_of_dicts = [
            {hyperparameters[idx]: [value] for idx, value in enumerate(default)} for default in defaults
        ]

        iris = sklearn.datasets.fetch_openml('iris', 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=1)

        fixture = 0.9555555555555556
        result = meta_model.evaluate(sklearn.svm.SVC(random_state=1),
                                     list_of_dicts[0:4], X_train, X_test, y_train, y_test)
        self.assertAlmostEqual(result, fixture)

if __name__ == '__main__':
    unittest.main()