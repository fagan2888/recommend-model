from sklearn.linear_model import LogisticRegression


class LR(LogisticRegression):

    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
        fit_intercept=True, intercept_scaling=1, class_weight=None,
        random_state=None, solver='liblinear', max_iter=100,
        multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        self.threshold = threshold

        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
            random_state=random_state, solver=solver, max_iter=max_iter,
            multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
         super(LR, self).fit(X, y, sample_weight=sample_weight)
         self.coef_old_ = self.coef_.copy()
         self.l2.fit(X, y, sample_weight=sample_weight)

         cntOfRow, cntOfCol = self.coef_.shape

         for i in range(cntOfRow):
             for j in range(cntOfCol):
                 coef = self.coef_[i][j]
                 if coef != 0:
                     idx = [j]
                     coef1 = self.l2.coef_[i][j]
                     for k in range(cntOfCol):
                         coef2 = self.l2.coef_[i][k]
                         if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                             idx.append(k)

                     mean = coef / len(idx)
                     self.coef_[i][idx] = mean

         return self
