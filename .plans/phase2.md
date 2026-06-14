# Phase 2: High-Value ML Additions ✅ COMPLETE

Core ML algorithms and evaluation tools missing from sklearn.

## sklearn.cluster

- [x] `DBSCAN` -- density-based clustering

## sklearn.ensemble

- [x] `GradientBoostingClassifier` -- gradient boosted trees (classifier)
- [x] `GradientBoostingRegressor` -- gradient boosted trees (regressor)

## sklearn.tree

- [x] `DecisionTreeRegressor` -- CART regression tree

## sklearn.metrics

- [x] `precision_score` -- precision (positive predictive value)
- [x] `recall_score` -- recall (sensitivity)
- [x] `f1_score` -- F1 score
- [x] `roc_auc_score` -- area under ROC curve
- [x] `roc_curve` -- ROC curve points
- [x] `mean_absolute_error` -- MAE
- [x] `precision_recall_curve` -- precision-recall curve
- [x] `log_loss` -- logistic loss / cross-entropy
- [x] `balanced_accuracy_score` -- balanced accuracy
- [x] `matthews_corrcoef` -- Matthews correlation coefficient
- [x] `cohen_kappa_score` -- Cohen's kappa

## sklearn.model_selection

- [x] `KFold` -- k-fold cross-validation splitter
- [x] `StratifiedKFold` -- stratified k-fold
- [x] `cross_val_score` -- cross-validation scoring (functional)
- [x] `GridSearchCV` -- grid search with cross-validation
- [ ] `RandomizedSearchCV` -- randomized hyperparameter search (deferred)

## sklearn.pipeline

- [x] `Pipeline` -- sequential feature transform + estimator
- [x] `make_pipeline` -- convenience constructor

## sklearn.impute

- [x] `SimpleImputer` -- fill missing with mean/median/mode/constant

## sklearn.svm

- [x] `SVC` -- support vector classifier (RBF, linear, poly kernels)
- [x] `SVR` -- support vector regressor
- [x] `LinearSVC` -- linear SVM (faster for linear kernel)

## sklearn.neural_network

- [x] `MLPClassifier` -- multi-layer perceptron classifier
- [x] `MLPRegressor` -- multi-layer perceptron regressor
