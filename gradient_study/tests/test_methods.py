import inspect
import unittest

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.methods import _kernel, fit_gd, learning_rate, mse, ols_statistics, r2_score_manual, select_features
from src.study import encode_train, experiment, prepare_loocv, split_dataset, TARGET


class OptimizerTests(unittest.TestCase):
    def test_learning_rates_start_at_zero(self):
        self.assertEqual(learning_rate(0, .1, 'time_decay'), .1)
        self.assertAlmostEqual(learning_rate(3, .1, 'time_decay'), .05)
        self.assertEqual(learning_rate(100, .1, 'constant'), .1)
        with self.assertRaises(ValueError):
            learning_rate(0, 0, 'constant')

    def test_full_updates_match_lecture_equations(self):
        X = np.array([[1., -1.], [1., 0.], [1., 2.]])
        y = np.array([0., 1., 3.])
        for method in [0, 3, 4]:
            with self.subTest(method=method):
                w, h, m, v = (np.zeros(2) for _ in range(4))
                for k in range(3):
                    eta = .1/np.sqrt(1+k)
                    g = 2*X.T@(X@w-y)/len(y)
                    if method == 0:
                        w -= eta*g
                    elif method == 3:
                        h = .9*h+eta*g
                        w -= h
                    else:
                        m = .9*m+.1*g
                        v = .999*v+.001*g*g
                        w -= eta*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8)
                actual, _, _, _ = _kernel(X,y,method,.1,True,3,0.,32,42,1,1e12)
                np.testing.assert_allclose(actual,w,rtol=1e-12,atol=1e-12)

    def test_sag_memory_and_average_match_reference(self):
        X = np.array([[1., -1.], [1., 0.], [1., 2.]])
        y = np.array([0., 1., 3.])
        rng = np.random.RandomState(42)
        w, memory, average = np.zeros(2), np.zeros((3,2)), np.zeros(2)
        for k in range(12):
            i = rng.randint(3)
            new = 2*(X[i]@w-y[i])*X[i]
            average += (new-memory[i])/3
            memory[i] = new
            np.testing.assert_allclose(average,memory.mean(axis=0),atol=1e-14)
            w -= .01/np.sqrt(1+k)*average
        actual, _, _, _ = _kernel(X,y,2,.01,True,12,0.,1,42,1,1e12)
        np.testing.assert_allclose(actual,w,rtol=1e-12,atol=1e-12)

    def test_sgd_batch_gradient_matches_reference(self):
        X = np.array([[1., -1.], [1., 0.], [1., 2.]])
        y = np.array([0., 1., 3.])
        rng = np.random.RandomState(42)
        w = np.zeros(2)
        for _ in range(4):
            indices = rng.randint(3,size=2)
            g = 2*X[indices].T@(X[indices]@w-y[indices])/2
            w -= .01*g
        actual, _, _, _ = _kernel(X,y,1,.01,False,4,0.,2,42,1,1e12)
        np.testing.assert_allclose(actual,w,rtol=1e-12,atol=1e-12)

    def test_convergence_and_original_scale(self):
        X = np.arange(30.).reshape(-1,1)
        y = 1200+300*X[:,0]
        for method, eta in [('VGD',.1),('SGD',.01),('SAG',.01),('Momentum',.01),('Adam',.05)]:
            model = fit_gd(X,y,method,eta,max_iter=10000,tolerance=1e-6)
            self.assertEqual(model.status,'converged',method)
            np.testing.assert_allclose(model.coefficients(),[1200,300],atol=.02)
            self.assertLess(mse(y,model.predict(X)),.001)

    def test_small_step_is_not_false_convergence(self):
        model = fit_gd(np.arange(20.).reshape(-1,1),np.arange(20.),initial=1e-12,max_iter=100)
        self.assertEqual(model.status,'max_iter')

    def test_divergence_is_exposed(self):
        model = fit_gd(np.arange(20.).reshape(-1,1),np.arange(20.),initial=10.,max_iter=100)
        self.assertEqual(model.status,'diverged')
        with self.assertRaises(ValueError):
            model.predict(np.zeros((1,1)))

    def test_statistics_match_independent_statsmodels(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(100,3)),columns=['a','b','c'])
        y = 300+X.to_numpy()@np.array([20.,0.,-5.])+rng.normal(size=100)
        table, model, params = ols_statistics(X,y)
        reference = sm.OLS(y,sm.add_constant(X)).fit()
        np.testing.assert_allclose(params,reference.params,rtol=1e-11)
        np.testing.assert_allclose(table['Стандартная ошибка'],reference.bse,rtol=1e-11)
        np.testing.assert_allclose(table['p_value'],reference.pvalues,atol=1e-12)
        self.assertAlmostEqual(model['F'],reference.fvalue,places=7)
        self.assertAlmostEqual(model['t_critical'],stats.t.ppf(.975,96))

    def test_metrics_and_constant_target(self):
        self.assertAlmostEqual(r2_score_manual(np.array([1.,2.,3.]),np.array([1.,2.,3.])),1.)
        with self.assertRaises(ValueError):
            r2_score_manual(np.ones(3),np.ones(3))
        with self.assertRaises(ValueError):
            fit_gd(np.array([[np.nan],[1.]]),np.array([1.,2.]))


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.data = pd.read_csv(Path(__file__).resolve().parents[1]/'data/regression_data.csv')

    def test_partitions_are_original_and_disjoint(self):
        a,b,c = split_dataset(self.data)
        self.assertEqual([len(a),len(b),len(c)],[480,160,160])
        self.assertEqual(len(set(a.index)|set(b.index)|set(c.index)),800)
        self.assertFalse(set(a.index)&set(b.index))
        self.assertFalse(set(a.index)&set(c.index))

    def test_feature_selection_is_computed_and_preserves_y(self):
        a,_,_ = split_dataset(self.data)
        X = encode_train(a.drop(columns=[TARGET]))
        y = a[TARGET].to_numpy()
        original_y = y.copy()
        features,_ = select_features(X,y)
        self.assertEqual(features,['Общая_площадь_м2','Этаж','Год_постройки','Расстояние_до_метро_мин','Номер_подъезда'])
        np.testing.assert_array_equal(y,original_y)

    def test_loocv_excludes_each_held_out_row(self):
        a,_,_ = split_dataset(self.data)
        sample = a.iloc[:60].copy()
        y = sample[TARGET].to_numpy()
        folds,stability = prepare_loocv(sample.drop(columns=[TARGET]),y,.05,.85)
        self.assertEqual(len(folds),60)
        for i,(X,fit_y,held_y,truth) in enumerate(folds):
            self.assertEqual(len(fit_y),59)
            self.assertEqual(truth,y[i])
            np.testing.assert_array_equal(fit_y,np.delete(y,i))
            self.assertEqual(held_y.shape,(1,X.shape[1]))
        self.assertTrue((stability['Доля folds']<=1).all())

    def test_search_interface_has_no_test_inputs(self):
        self.assertNotIn('test',inspect.signature(experiment).parameters)


if __name__ == '__main__':
    unittest.main()
