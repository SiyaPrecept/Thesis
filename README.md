======================================
Random forest model
======================================
1. Data Preparation
  Loads and renames variables from input datasets (e.g., RF_348_TXT.txt for wards).
  Creates spatial clusters using KMeans (configurable n_clusters) based on longitude/latitude for holdout and stratified cross-validation.
  Selects base features (e.g., DomesticFuelUse, PopDensity) plus target-specific NDVI; handles missing columns gracefully.
2. Modeling Approach
  Trains Random Forest Regressors separately for each season (winter, spring, summer, autumn) and annual targets (PM_2km, PM_6km).
  Creates 90/10 spatial holdout via whole-cluster selection (fallback to random split if no coords).
  Uses spatial K-fold CV (or standard KFold) on non-holdout data.
  Performs GridSearchCV hyperparameter tuning over estimators, depth, samples, and features (configurable grid).
3. Evaluation Metrics
  Computes R² and Root Mean Squared Error (RMSE) on CV folds, holdout test set, and full data.
  Saves metrics to JSON/CSV (e.g., best_params.json, holdout_meta.json).
4. Feature Importance & Explainability
  Calculates Gini importance on full model and Permutation importance on test set.
  Generates Partial Dependence Plots (PDPs) and Individual Conditional Expectation (ICE) plots with rugs for feature–response relationships.
  Combines PDPs/ICE into summary visualizations and CSVs across targets (e.g., all_features_combined_pdp.png).
5. Learning and Performance Diagnostics
  Produces learning curves (train/validation RMSE vs data fraction) to assess generalization.
  Creates scatter plots for test/full predictions (individual and combined subplots).
  Generates spatial cluster visualizations with concentration bins (PNG/CSV).
6. Outputs
  Model diagnostics: scatter plots, feature importances, PDPs/ICE CSVs/PNGs (individual/combined).
  Cluster visualization: spatial_clusters_with_concentration.png/CSV.
  Tabular outputs:
    combined_test_predictions_all_targets.csv
    combined_full_predictions_all_targets.csv
    permutation_importance_test.csv
    gini_importance_full.csv
    combined_pdp_data.csv
    combined_ice_data_all_targets.csv
    combined_learning_curves.csv
    rf_overall_summary.csv


======================================
Multi-layer perceptron neural network
======================================

1. Data Preparation
  Loads and processes data from tab-delimited TXT files (e.g., VARS_348_2.txt for wards dataset).
  Applies log transformation to targets if enabled (USE_LOG_TARGET); handles NaNs with mean imputation.
  Selects base features (e.g., BuiltUpDensity, Elevation, NDVI seasonal/annual) per target.
  Uses spatial coordinates (Lon/Lat) for KMeans-based clustering to create holdout set (10%) and stratified CV folds.
2. Modeling Approach
  Trains custom MLP (PMPredictor) separately for each season (winter, spring, summer, autumn) and annual targets (PM_2km, PM_6km).
  Performs hyperparameter tuning via Optuna (layers, hidden size, dropout, learning rate, weight decay) with configurable trials and search spaces.
  Uses spatial K-fold CV on 90% data (fallback to standard KFold if no coords); weighted sampling for imbalanced targets.
  Trains with AdamW optimizer, custom Huber+bias loss, mixed-precision (AMP), gradient clipping, early stopping, and LR plateau scheduler.
  Final model trained on full non-holdout data with capacity constraints (e.g., max hidden size).
  3. Evaluation Metrics
  Computes R² and Root Mean Squared Error (RMSE) on CV folds, full trainval, and untouched holdout set.
  Applies optional prediction calibration using quantile-based error correction.
  Saves metrics to JSON/CSV (e.g., meta_fold_*.json, holdout_results.json).
4. Feature Importance & Explainability
  Calculates Permutation importance for each feature on final model.
  Generates Partial Dependence Plots (PDPs) for feature-response relationships (grid resolution=50).
  Computes SHAP values (DeepExplainer preferred, fallback to KernelExplainer) on sampled data; saves summaries and plots.
Combines PDPs and importances across targets into unified CSVs and visualizations.
5. Learning and Performance Diagnostics
  Produces learning curves (train/val loss per fold; final train loss).
  Creates prediction bias plots (mean error vs true value bins) per fold.
  Generates spatial cluster visualizations with concentration bins (PNG/CSV).
  Combines learning curves and scatter plots (CV/full/holdout predictions) across targets.
6. Outputs
  Model artifacts: .pth files (per fold and final), scalers (.npy).
  Diagnostics: scatter plots, bias plots, feature importances/PDPs/SHAP CSVs/PNGs.
  Cluster maps: spatial_clusters_with_concentration.png/CSV.
  Tabular outputs:
  holdout_predictions.csv
  full_predictions.csv
  permutation_importance.csv
  combined_pdp_all_targets.csv
  combined_test_predictions.csv
  combined_learning_curves.csv
  summary_results.csv



