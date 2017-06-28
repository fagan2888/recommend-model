# coding = utf8
from __future__ import division
import pandas as pd
import numpy as np
import warnings

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
#from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings('ignore')


class Svm(object):
    def __init__(self, path, test_num):
        self.path = path
        self.data = pd.read_csv(path, index_col = 0, parse_dates = True)
        self.cv = int(round(len(self.data)/50.0))
#        self.window = len(self.data) - test_num - 10
        self.window = 300
        if self.window + test_num > len(self.data):
            self.window = 150

        self.feature = self.preprocessing()
        self.threshold = self.get_threshold(test_num)
        self.score, self.params, self.feature_selected = self.pipelining(test_num, threads_num = 32)
        self.result_df, self.win_ratio = self.training(test_num)
        self.threshold_value, self.invest_value = self.simulating(test_num)


    def preprocessing(self):
        feature = self.data
        pct_chg = feature['pct_chg']
        feature.rename(columns = {'pct_chg': 'pct_chg_f'}, inplace = True)

#        feature = feature.apply(MinMaxScaler().fit_transform)
        feature = feature.apply(scale)
        feature_lag1 = feature.shift(1)
        feature_lag2 = feature.shift(2)
        feature_lag3 = feature.shift(3)

        feature_lag1.rename(columns = lambda x: (x + '_lag1'), inplace = True)
        feature_lag2.rename(columns = lambda x: (x + '_lag2'), inplace = True)
        feature_lag3.rename(columns = lambda x: (x + '_lag3'), inplace = True)
        feature = pd.concat([feature, feature_lag1, feature_lag2, feature_lag3], axis = 1)

        feature['pct_chg'] = pct_chg
        feature['label'] = np.sign(feature['pct_chg'])
        feature['label'] = feature['label'].shift(-1)
        feature.dropna(inplace = True)
#        print feature.columns

        return feature


    def get_threshold(self, test_num):
        correct_num = (np.sign(self.feature['pct_chg']) == self.feature['label'])[-test_num:].sum()
        win_ratio = correct_num/test_num

        return win_ratio


    def pipelining(self, test_num, threads_num):

        score_list = []
        params_list = []
        feature_selected_list = []

        for lag in range(4):

            if lag == 0:
                x = self.feature.loc[:, 'close':'dpo']
            elif lag == 1:
                x = self.feature.loc[:, 'close':'dpo_lag1']
            elif lag == 2:
                x = self.feature.loc[:, 'close':'dpo_lag2']
            elif lag == 3:
                x = self.feature.loc[:, 'close':'dpo_lag3']


            y = self.feature['label']
            cv = self.cv

            selectkbest = SelectKBest(f_regression)
            svc = SVC()
            pipe = Pipeline(steps = [('selectkbest', selectkbest), ('svc', svc)])

            params = dict(
                    selectkbest__k = range(1, 6),
                    svc__C = 10.0**np.arange(-5, 5),
                    svc__gamma = 10.0**np.arange(-5, 5)
                    )

            estimator = GridSearchCV(pipe, param_grid = params, cv = cv, verbose = 0, n_jobs = threads_num, scoring = 'accuracy')
            estimator.fit(x, y)
            score = estimator.best_score_
            params = estimator.best_params_
            feature_selected = x.columns[estimator.best_estimator_.named_steps['selectkbest'].get_support()]

            score_list.append(score)
            params_list.append(params)
            feature_selected_list.append(feature_selected)

        best_lag = np.argmax(score_list)
        score  = score_list[best_lag]
        params = params_list[best_lag]
        feature_selected = feature_selected_list[best_lag]

        return score, params, feature_selected


    # c: svm__C, g: svm__gamma
    def training(self, test_num):
        window = self.window
        c = self.params['svc__C']
        g = self.params['svc__gamma']
        x = self.feature.loc[:, self.feature_selected]
        y = self.feature['label']

        pre_states = []
        for i in np.arange(test_num):
            x_train, x_test = x[-test_num - window + i: -test_num + i], x[-test_num + i: ]
            y_train = y[-test_num - window + i: -test_num + i]

            svc = SVC(C = c, gamma = g)
            svc.fit(x_train, y_train)
            pre_state = svc.predict(x_test)[0]
            pre_states.append(pre_state)

        pre_states = np.array(pre_states)

        result_df = self.feature[-test_num: ]
        result_df.loc[:, 'pre_states'] = pre_states

        correct_num = (result_df['pre_states'] == result_df['label']).values.sum()
        win_ratio = correct_num/test_num

        return result_df, win_ratio


    def simulating(self, test_num):
        dates = self.result_df.index
        threshold_value = 1.0
        invest_value = 1.0

        for i in range(test_num - 1):
            if self.result_df.loc[dates[i], 'pct_chg'] > 0:
                threshold_value *= (1 + self.result_df.loc[dates[i+1], 'pct_chg']/100)
#            else:
#                threshold_value *= (1 - self.result_df.loc[dates[i+1], 'pct_chg']/100)

            if self.result_df.loc[dates[i], 'pre_states'] > 0:
                invest_value *= (1 + self.result_df.loc[dates[i+1], 'pct_chg']/100)
#            else:
#                invest_value *= (1 - self.result_df.loc[dates[i+1], 'pct_chg']/100)

        return threshold_value, invest_value


if __name__ == '__main__':

    test_num = 250
    assets = ['sh300', 'zz500', 'hsi', 'nhsp', 'sp500', 'au']

    for asset in assets:
        svm = Svm('../assets/' + asset + '.csv', test_num)
        print 'asset: ', asset
        #print 'window: ',svm.window
        #print 'cv: ', svm.cv
        print 'threshold: ', svm.threshold
        print 'win_ratio: ', svm.win_ratio
        #print 'cv_win_ratio: ', svm.score
        print 'params: ', svm.params
        print 'features: ', svm.feature_selected
        #print 'threshold_value: ', svm.threshold_value
        #print 'invest_value: ', svm.invest_value
        print
