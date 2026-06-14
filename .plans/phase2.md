# Phase 2: High-Value ML Additions

Core ML algorithms and evaluation tools missing from sklearn.

## sklearn.cluster

- [ ] `DBSCAN` -- density-based clustering

## sklearn.ensemble

- [ ] `GradientBoostingClassifier` -- gradient boosted trees (classifier)
- [ ] `GradientBoostingRegressor` -- gradient boosted trees (regressor)

## sklearn.tree

- [ ] `DecisionTreeRegressor` -- CART regression tree

## sklearn.metrics

- [ ] `precision_score` -- precision (positive predictive value)
- [ ] `recall_score` -- recall (sensitivity)
- [ ] `f1_score` -- F1 score
- [ ] `roc_auc_score` -- area under ROC curve
- [ ] `roc_curve` -- ROC curve points
- [ ] `mean_absolute_error` -- MAE
- [ ] `precision_recall_curve` -- precision-recall curve
- [ ] `log_loss` -- logistic loss / cross-entropy
- [ ] `balanced_accuracy_score` -- balanced accuracy
- [ ] `matthews_corrcoef` -- Matthews correlation coefficient
- [ ] `cohen_kappa_score` -- Cohen's kappa

## sklearn.model_selection

- [ ] `KFold` -- k-fold cross-validation splitter
- [ ] `StratifiedKFold` -- stratified k-fold
- [ ] `cross_val_score` -- cross-validation scoring (functional)
- [ ] `GridSearchCV` -- grid search with cross-validation
- [ ] `RandomizedSearchCV` -- randomized hyperparameter search

## sklearn.pipeline

- [ ] `Pipeline` -- sequential feature transform + estimator
- [ ] `make_pipeline` -- convenience constructor

## sklearn.impute

- [ ] `SimpleImputer` -- fill missing with mean/median/mode/constant

## sklearn.svm

- [ ] `SVC` -- support vector classifier (RBF, linear, poly kernels)
- [ ] `SVR` -- support vector regressor
- [ ] `LinearSVC` -- linear SVM (faster for linear kernel)

## sklearn.neural_network

- [ ] `MLPClassifier` -- multi-layer perceptron classifier
- [ ] `MLPRegressor` -- multi-layer perceptron regressor
