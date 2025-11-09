import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, GroupKFold, learning_curve
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# === Configuration ===
SEASONS = ['winter', 'spring', 'summer', 'autumn']
ANNUAL_TARGETS = ['PM_2km', 'PM_6km']
BASE_FEATURES = [
    'Household Fuel Consumption', 'Population Density', 'Informal Settlements Density',
    'Built-Up Area Density', 'Industrial Area Density', 'Road Length', 'Elevation'
]
COORD_COLS = ['longitude', 'latitude']
COLUMN_NAMES = {
    'Lon': 'longitude',
    'Lat': 'latitude',
    'DomesticFuelUse': 'Household Fuel Consumption',
    'PopDensity': 'Population Density',
    'InformalDensity': 'Informal Settlements Density',
    'BuiltUpDensity': 'Built-Up Area Density',
    'IndurialDensity': 'Industrial Area Density',
    'RoadLength': 'Road Length',
    'Elevation': 'Elevation',
    'NDVIWinter': 'NDVI (Winter)',
    'NDVISpring': 'NDVI (Spring)',
    'NDVISummer': 'NDVI (Summer)',
    'NDVIAutumn': 'NDVI (Autumn)',
    'NDVI_annual': 'NDVI (Annual)',
    'winter': 'Winter CAMx PM₂.₅',
    'spring': 'Spring CAMx PM₂.₅',
    'summer': 'Summer CAMx PM₂.₅',
    'autumn': 'Autumn CAMx PM₂.₅'
}

metrics_results = []
all_grid_results = []

OUTPUT_DIR = '/home/siya/test_1/RF_strat/output_2/Wards_run150/_2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAME_MAP = {
    'Household Fuel Consumption': 'Domestic fuel use',
    'Built-Up Area Density': 'Built-up area density',
    'Population Density': 'Population density',
    'Informal Settlements Density': 'Informal settlements',
    'Industrial Area Density': 'Industrial areas',
    'Road Length': 'Distance to roads',
    'Elevation': 'Elevation',
    'NDVI': 'Vegetation index (NDVI)'
}

# Load and prepare data
df = pd.read_csv('/home/siya/test_1/RF_strat/TXT/RF_348_TXT.txt', sep='\t')
df.rename(columns=COLUMN_NAMES, inplace=True)

# Debug: Print columns to verify coordinate names
print("Columns after renaming:", df.columns.tolist())

# Create spatial clusters
coords_found = False
if all(col in df.columns for col in COORD_COLS):
    coords = df[COORD_COLS].values
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    df['spatial_cluster'] = kmeans.fit_predict(coords)
    coords_found = True
    print("Spatial clusters created using coordinates")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(df['longitude'], df['latitude'], c=df['spatial_cluster'], cmap='tab20', s=50)
    plt.colorbar(label='Cluster ID')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Clusters of Wards')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/spatial_clusters_map.png")
    plt.close()
else:
    print("Warning: Coordinate columns not found. Using random CV instead.")
    df['spatial_cluster'] = np.arange(len(df))

df[['WARD_ID', 'spatial_cluster', 'longitude', 'latitude']].to_csv(
    f"{OUTPUT_DIR}/spatial_cluster_assignments.csv", index=False
)

# Remove highly correlated features (|r| > 0.9)
corr_matrix = df[BASE_FEATURES].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
BASE_FEATURES = [f for f in BASE_FEATURES if f not in to_drop]
print(f"Removed correlated features: {to_drop}")

# Initialize results storage
metrics_results = []
gini_results = []
perm_results = []
scatter_data = []
cv_data = []
pdp_storage = {}

# NEW: Define custom RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Parameter grid with increased n_estimators
param_grid = {
    'n_estimators': list(range(50, 151, 1)),  # Increased from [10]
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rmse_scorer = make_scorer(rmse, greater_is_better=False)
r2_scorer = make_scorer(r2_score)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

r2_vs_trees_data = {}

def plot_learning_curve(estimator, title, X, y, output_path,
                        cv=KFold(n_splits=10, shuffle=True, random_state=42),
                        train_sizes=np.linspace(0.1, 1.0, 10), groups=None):
    """Plot learning curve with RMSE"""
    # NEW: Use custom RMSE scorer
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        groups=groups,
        scoring=rmse_scorer,
        shuffle=True,
        random_state=42,
        n_jobs=-1,
        return_times=False
    )

    # NEW: Scores are already RMSE due to custom scorer
    train_rmse = -train_scores
    valid_rmse = -valid_scores

    records = []
    for i, frac in enumerate(train_sizes):
        for fold in range(train_rmse.shape[1]):
            records.append({
                'train_fraction': frac,
                'fold': fold + 1,
                'train_rmse': train_rmse[i, fold],
                'valid_rmse': valid_rmse[i, fold]
            })
    pd.DataFrame(records).to_csv(f"{output_path}_learning_curve_data.csv", index=False)

    train_mean = train_rmse.mean(axis=1)
    train_std = train_rmse.std(axis=1)
    valid_mean = valid_rmse.mean(axis=1)
    valid_std = valid_rmse.std(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='tab:blue', label='Training RMSE')
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.2, color='tab:blue')
    plt.plot(train_sizes, valid_mean, 's-', color='tab:orange', label='Validation RMSE')
    plt.fill_between(train_sizes,
                     valid_mean - valid_std,
                     valid_mean + valid_std,
                     alpha=0.2, color='tab:orange')

    plt.title(title, fontsize=14)
    plt.xlabel('Fraction of Training Data', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.xticks(train_sizes, [f'{s:.1f}' for s in train_sizes], fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_path}_learning_curve.png", dpi=300)
    plt.close()

    return train_mean, valid_mean

def compute_and_store_pdp(model, X, features, model_name):
    for feature in features:
        storage_key = 'NDVI' if 'NDVI' in feature else feature
        pdp = partial_dependence(
            model, X, features=[feature],
            kind='average', grid_resolution=50
        )
        # NEW: Handle different key names based on scikit-learn version
        values = pdp.get('values', pdp.get('grid_values'))[0]
        avg_prediction = pdp['average'][0]
        if storage_key not in pdp_storage:
            pdp_storage[storage_key] = []
        pdp_storage[storage_key].append({
            'model': model_name,
            'values': values,
            'average': avg_prediction
        })

def evaluate_model(X, y, label, spatial_clusters):
    # NEW: Check data quality
    print(f"Checking data for {label}:")
    print("X has nan:", X.isna().any().any())
    print("X has inf:", np.isinf(X).any().any())
    print("y has nan:", y.isna().any())
    print("y has inf:", np.isinf(y).any())
    print("y variance:", y.var())
    
    # NEW: Remove nan or inf values and clip negative PM₂.₅
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index].clip(lower=0)  # Ensure non-negative PM₂.₅
    spatial_clusters = spatial_clusters[X.index]
    
    # NEW: Warn if variance is too low
    if y.var() < 1e-10:
        print(f"Warning: Low variance in target {label}: {y.var()}")

    model_name_clean = label.replace(" ", "_")
    model_dir = os.path.join(OUTPUT_DIR, model_name_clean)
    os.makedirs(model_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, clusters_train, clusters_test = train_test_split(
        X, y, spatial_clusters, test_size=0.1, random_state=42
    )

    if coords_found and len(np.unique(clusters_train)) > 1:
        cv = GroupKFold(n_splits=5)
        cv_groups = clusters_train
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_groups = None

    model = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        param_grid,
        cv=cv,
        scoring={'rmse': rmse_scorer, 'r2': r2_scorer},
        refit='rmse',
        return_train_score=True,
        n_jobs=8,
        verbose=1
    )
    if cv_groups is not None:
        model.fit(X_train, y_train, groups=cv_groups)
    else:
        model.fit(X_train, y_train)

    best_model = model.best_estimator_
    y_test_pred = best_model.predict(X_test)
    train_r2 = r2_score(y_train, best_model.predict(X_train))
    test_r2 = r2_score(y_test, y_test_pred)
    # NEW: Use custom RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    metrics_results.append({
        'Model': label,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Best_Params': str(model.best_params_)
    })

    cv_df = pd.DataFrame(model.cv_results_)
    records = []
    for _, row in cv_df.iterrows():
        n_est = row['param_n_estimators']
        mf = row['param_max_features']
        md = row['param_max_depth']
        mss = row['param_min_samples_split']
        msl = row['param_min_samples_leaf']
        for fold in range(model.n_splits_):
            records.append({
                'Model': label,
                'n_estimators': n_est,
                'max_features': mf,
                'max_depth': md,
                'min_samples_split': mss,
                'min_samples_leaf': msl,
                'fold': fold,
                'Train_R2': row[f'split{fold}_train_r2'],
                'Test_R2': row[f'split{fold}_test_r2'],
                'Train_RMSE': -row[f'split{fold}_train_rmse'],
                'Test_RMSE': -row[f'split{fold}_test_rmse']
            })

    all_grid_results.append(pd.DataFrame(records))

    scatter_df = pd.DataFrame({
        'Observed': y_test,
        'Predicted': y_test_pred
    }, index=y_test.index)
    if 'longitude' in df.columns and 'latitude' in df.columns:
        scatter_df['longitude'] = df.loc[scatter_df.index, 'longitude'].to_numpy()
        scatter_df['latitude'] = df.loc[scatter_df.index, 'latitude'].to_numpy()

    scatter_df.to_csv(f"{model_dir}/scatter_data.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.scatter(scatter_df['Observed'], scatter_df['Predicted'], alpha=0.6)
    plt.plot(
        [scatter_df['Observed'].min(), scatter_df['Observed'].max()],
        [scatter_df['Observed'].min(), scatter_df['Observed'].max()],
        '--r'
    )
    plt.xlabel('Observed PM₂.₅')
    plt.ylabel('Predicted PM₂.₅')
    plt.title(f'{label} – Test Set Performance\nR² = {test_r2:.3f}, RMSE = {test_rmse:.3f}')
    plt.grid(True)
    plt.savefig(f'{model_dir}/scatter_plot.png')
    plt.close()

    perm = permutation_importance(best_model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
    for feat, gini, perm_imp in zip(X.columns, best_model.feature_importances_, perm.importances_mean):
        gini_results.append({'Model': label, 'Feature': feat, 'Gini': gini})
        perm_results.append({'Model': label, 'Feature': feat, 'Permutation': perm_imp})
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Gini': best_model.feature_importances_,
        'Permutation': perm.importances_mean
    })
    feature_importance.to_csv(f"{model_dir}/feature_importance.csv", index=False)

    plot_learning_curve(
        best_model,
        label,
        X_train,
        y_train,
        output_path=f"{model_dir}/{label}",
        cv=cv,
        groups=cv_groups
    )

    compute_and_store_pdp(best_model, X_train, X.columns, label)

    r2_results = []
    cv_results = pd.DataFrame(model.cv_results_)
    for n_est in param_grid['n_estimators']:
        mask = cv_results['param_n_estimators'] == n_est
        if not mask.any():
            continue
        r2_scores = []
        for i in range(model.n_splits_):
            fold_col = f'split{i}_test_r2'
            if fold_col in cv_results.columns:
                r2_scores.extend(cv_results.loc[mask, fold_col].values)
        if r2_scores:
            mean_r2 = np.mean(r2_scores)
            std_r2 = np.std(r2_scores)
            r2_results.append({
                'n_estimators': n_est,
                'mean_r2': mean_r2,
                'std_r2': std_r2
            })
    
    r2_vs_trees_data[model_name_clean] = pd.DataFrame(r2_results)
    
    cv_info = []
    for fold_idx, (train_index, test_index) in enumerate(model.cv.split(X_train, y_train, groups=cv_groups)):
        for idx in train_index:
            cv_info.append({
                'model': label,
                'fold': fold_idx,
                'data_point': X_train.index[idx],
                'set': 'train',
                'spatial_cluster': clusters_train.iloc[idx] if coords_found else -1
            })
        for idx in test_index:
            cv_info.append({
                'model': label,
                'fold': fold_idx,
                'data_point': X_train.index[idx],
                'set': 'validation',
                'spatial_cluster': clusters_train.iloc[idx] if coords_found else -1
            })
    cv_data.extend(cv_info)

def generate_combined_pdp_plot(pdp_storage, output_dir):
    os.makedirs(f'{output_dir}/combined_pdp', exist_ok=True)
    model_colors = {
        'Winter_Seasonal': '#A80000',
        'Spring_Seasonal': '#FF5B04',
        'Summer_Seasonal': '#00A884',
        'Autumn_Seasonal': '#0070FF',
        'Annual_PM_2km': 'black',
        'Annual_PM_6km': '#E60000'
    }
    line_styles = {
        'Winter_Seasonal': '-',
        'Spring_Seasonal': '-',
        'Summer_Seasonal': '-',
        'Autumn_Seasonal': '-',
        'Annual_PM_2km': '--',
        'Annual_PM_6km': '--'
    }
    model_display_names = {
        'Winter_Seasonal': 'Winter',
        'Spring_Seasonal': 'Spring',
        'Summer_Seasonal': 'Summer',
        'Autumn_Seasonal': 'Autumn',
        'Annual_PM_6km': 'Annual',
        'Annual_PM_2km': '2 km annual'

    }

    for feature, data_list in pdp_storage.items():
        plt.figure(figsize=(10, 6))
        for data in data_list:
            model_name = data['model']
            display_name = model_display_names.get(model_name, model_name)
            plt.plot(
                data['values'], data['average'],
                label=display_name,
                color=model_colors.get(model_name, 'gray'),
                linestyle=line_styles.get(model_name, '-'),
                linewidth=2.5
            )
        feature_title = FEATURE_NAME_MAP.get(feature, feature)
        plt.title(f'Partial Dependence: {feature_title}', fontsize=14)
        plt.xlabel('Normalized Feature Value', fontsize=13)
        plt.ylabel(r'PM$_{2.5}$ concentration ($\mu$g$\cdot$m$^{-3}$)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=14)
        plt.xticks(np.linspace(0, 1, 11), [f'{x:.1f}' for x in np.linspace(0, 1, 11)], fontsize=12)
        plt.yticks(fontsize=12)
        plt.gca().tick_params(axis='y', which='both', left=True, labelleft=True)
        safe_feature_name = feature.replace(' ', '_').replace('/', '_')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/combined_pdp/{safe_feature_name}_pdp.png', dpi=300)
        plt.close()

    all_features = sorted(pdp_storage.keys())
    n_cols = 3
    n_rows = (len(all_features) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharey=False)
    legend_added = False
    for idx, feature in enumerate(all_features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axs[row, col] if n_rows > 1 else axs[col]
        lines, labels = [], []
        for data in pdp_storage[feature]:
            model_name = data['model']
            display_name = model_display_names.get(model_name, model_name)
            line = ax.plot(
                data['values'], data['average'],
                color=model_colors.get(model_name, 'gray'),
                linestyle=line_styles.get(model_name, '-'),
                linewidth=2
            )
            lines.append(line[0])
            labels.append(display_name)
        feature_title = FEATURE_NAME_MAP.get(feature, feature)
        ax.set_title(feature_title, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 11)], fontsize=12)
        ax.set_ylim(20, 80)
        ax.set_yticks(np.arange(20, 81, 10))
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='y', which='both', left=True, labelleft=True)
        if col == 0:
            ax.set_ylabel(r'PM$_{2.5}$ concentration ($\mu$g$\cdot$m$^{-3}$)', fontsize=14)
        if feature_title == "Domestic fuel use" and not legend_added:
            ax.legend(lines, labels, loc='best', fontsize=12)
            legend_added = True

    total_plots = n_rows * n_cols
    if len(all_features) < total_plots:
        for idx in range(len(all_features), total_plots):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                fig.delaxes(axs[row, col])
            else:
                fig.delaxes(axs[col])

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(f'{output_dir}/combined_pdp/all_features_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_pdp_to_csv(pdp_storage, output_dir):
    pdp_rows = []
    for feature, data_list in pdp_storage.items():
        for data in data_list:
            for x_val, y_val in zip(data['values'], data['average']):
                pdp_rows.append({
                    'Feature': feature,
                    'Model': data['model'],
                    'X_Value': x_val,
                    'Y_Value': y_val
                })
    pdp_df = pd.DataFrame(pdp_rows)
    pdp_df.to_csv(f'{output_dir}/pdp_values.csv', index=False)

def plot_r2_vs_trees(r2_vs_trees_data, output_dir):
    if not r2_vs_trees_data:
        print("Warning: No R² vs trees data available. Skipping plot.")
        return
    plt.figure(figsize=(12, 8))
    model_styles = {
        'Winter_Seasonal': {'color': '#A80000', 'marker': 'o', 'label': 'Winter'},
        'Spring_Seasonal': {'color': '#FF5B04', 'marker': 's', 'label': 'Spring'},
        'Summer_Seasonal': {'color': '#00A884', 'marker': '^', 'label': 'Summer'},
        'Autumn_Seasonal': {'color': '#0070FF', 'marker': 'D', 'label': 'Autumn'},
        'Annual_PM_2km': {'color': 'black', 'marker': 'X', 'label': '2 km Annual'},
        'Annual_PM_6km': {'color': '#E60000', 'marker': 'P', 'label': '6 km Annual'}
    }
    all_r2_values = []
    valid_models = False
    for model_name, df_data in r2_vs_trees_data.items():
        if model_name not in model_styles:
            continue
        if df_data.empty or len(df_data) < 2:
            print(f"Warning: Insufficient data for {model_name} in R² vs trees plot. Skipping.")
            continue
        valid_models = True
        style = model_styles[model_name]
        n_estimators = df_data['n_estimators']
        mean_r2 = df_data['mean_r2']
        std_r2 = df_data['std_r2']
        all_r2_values.extend(mean_r2 + std_r2)
        all_r2_values.extend(mean_r2 - std_r2)
        base_color = mcolors.to_rgba(style['color'])
        error_color = (base_color[0], base_color[1], base_color[2], 0.5)
        plt.errorbar(
            n_estimators, mean_r2, yerr=std_r2,
            fmt=style['marker'] + '-',
            color=style['color'],
            ecolor=error_color,
            elinewidth=2,
            capsize=5,
            capthick=2,
            label=style['label']
        )
    if not valid_models:
        print("No valid models with sufficient data for R² vs trees plot.")
        plt.close()
        return
    y_min = max(0.0, min(all_r2_values) - 0.05)
    y_max = min(1.0, max(all_r2_values) + 0.05)
    plt.ylim(y_min, y_max)
    plt.title('Cross-validated R² vs. Number of Trees', fontsize=16)
    plt.xlabel('Number of Trees', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/r2_vs_trees_cross_validated.png", dpi=300)
    plt.close()
    combined_data = []
    for model_name, df_data in r2_vs_trees_data.items():
        for _, row in df_data.iterrows():
            combined_data.append({
                'Model': model_name,
                'n_estimators': row['n_estimators'],
                'mean_r2': row['mean_r2'],
                'std_r2': row['std_r2']
            })
    pd.DataFrame(combined_data).to_csv(f"{output_dir}/r2_vs_trees_data.csv", index=False)

# === Main Execution ===
print("\n" + "="*50)
print("STARTING MODEL TRAINING PROCESS")
print("="*50 + "\n")

# Seasonal Models
print("Training seasonal models:")
for season in SEASONS:
    print(f"\n=== Starting {season.capitalize()} model ===")
    y_col = f"{season.capitalize()} CAMx PM₂.₅"
    ndvi_col = f"NDVI ({season.capitalize()})"
    features = BASE_FEATURES + [ndvi_col]
    sub = df[[y_col] + features + ['spatial_cluster']].dropna()
    X, y = sub[features], sub[y_col]
    spatial_clusters = sub['spatial_cluster']
    evaluate_model(X, y, f"{season.capitalize()}_Seasonal", spatial_clusters)
    print(f"=== Completed {season.capitalize()} model ===\n")

# Annual Models
print("\nTraining annual models:")
for target in ANNUAL_TARGETS:
    print(f"\n=== Starting {target} model ===")
    y_col = target
    features = BASE_FEATURES + ['NDVI (Annual)']
    sub = df[[y_col] + features + ['spatial_cluster']].dropna()
    X, y = sub[features], sub[y_col]
    spatial_clusters = sub['spatial_cluster']
    evaluate_model(X, y, f"Annual_{target}", spatial_clusters)
    print(f"=== Completed {target} model ===\n")

# Save summary outputs
pd.DataFrame(metrics_results).to_csv(f'{OUTPUT_DIR}/model_performance_metrics.csv', index=False)
full_df = pd.concat(all_grid_results, ignore_index=True)
full_df.to_csv(f"{OUTPUT_DIR}/fold_level_results.csv", index=False)

if gini_results:
    gini_df = pd.DataFrame(gini_results).pivot(index='Feature', columns='Model', values='Gini')
    gini_df['Average'] = gini_df.mean(axis=1)
    gini_df.to_csv(f'{OUTPUT_DIR}/gini_importance_comparison.csv', index=True)

if perm_results:
    perm_df = pd.DataFrame(perm_results).pivot(index='Feature', columns='Model', values='Permutation')
    perm_df['Average'] = perm_df.mean(axis=1)
    perm_df.to_csv(f'{OUTPUT_DIR}/permutation_importance_comparison.csv', index=True)

if cv_data:
    pd.DataFrame(cv_data).to_csv(f'{OUTPUT_DIR}/cv_split_details.csv', index=False)

if pdp_storage:
    print("\nGenerating PDP plots...")
    generate_combined_pdp_plot(pdp_storage, OUTPUT_DIR)
    save_pdp_to_csv(pdp_storage, OUTPUT_DIR)

plot_r2_vs_trees(r2_vs_trees_data, OUTPUT_DIR)

print("\n" + "="*50)
print("ALL PROCESSES COMPLETED SUCCESSFULLY!")
print("="*50)