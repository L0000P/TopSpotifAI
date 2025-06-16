import os
import glob
import json
import gc
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, 
                                   StandardScaler, MinMaxScaler)
from torch.utils.tensorboard import SummaryWriter

def extract_country_name(filename):
    """Extract clean country name from filename"""
    
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    if 'spotify-streaming-top-50-' in base_name:
        country = base_name.replace('spotify-streaming-top-50-', '')
    elif 'top-50-' in base_name:
        country = base_name.split('top-50-')[-1]
    else:
        country = base_name.split('-')[-1]
    
    country = country.replace('-', ' ').replace('_', ' ')
    country = ' '.join([word.capitalize() for word in country.split()])
    
    return country

def add_basic_features(df):
    """Add basic features without lag features or MLP-specific preprocessing"""
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["year"] = df["date"].dt.year
    
    # Cyclical encoding for temporal features
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
    
    # Release date features
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["days_since_release"] = (df["date"] - pd.to_datetime(df["release_date"], errors="coerce")).dt.days
    df["weeks_since_release"] = df["days_since_release"] / 7
    df["months_since_release"] = df["days_since_release"] / 30.44
    
    # For position prediction: popularity-based features
    df["popularity_squared"] = df["popularity"] ** 2
    df["popularity_log"] = np.log1p(df["popularity"])

    # For popularity prediction: position-based features
    df["position_squared"] = df["position"] ** 2
    df["position_log"] = np.log1p(df["position"])
    
    # Duration-based features
    df["duration_minutes"] = df["duration_ms"] / 60000
    df["duration_log"] = np.log1p(df["duration_ms"])
    
    # Categorical features
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday_season"] = ((df["month"] == 12) | (df["month"] == 1)).astype(int)
    df["is_summer"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)
    
    return df

def split_data_by_country(df, songs_per_day=50):
    """Split data by country with temporal train/test split"""
    
    train_data, test_data = [], []
    
    for country in tqdm(df["country"].unique(), desc="Processing countries"):
        country_mask = df["country"] == country
        cdf = df[country_mask].copy()
        
        cdf = cdf.sort_values(["date", "position"]).reset_index(drop=True)
        cdf_daily = cdf.groupby("date").head(songs_per_day).reset_index(drop=True)
        
        all_dates = sorted(cdf_daily["date"].unique())
        n_dates = len(all_dates)
        train_split = int(n_dates * 0.8)
        
        train_dates = set(all_dates[:train_split])
        test_dates = set(all_dates[train_split:])
        
        train_mask = cdf_daily["date"].isin(train_dates)
        test_mask = cdf_daily["date"].isin(test_dates)
        
        train_data.append(cdf_daily[train_mask])
        test_data.append(cdf_daily[test_mask])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    return train_df, test_df

def prepare_features_train_only(train_df):
    """Prepare features for both targets"""
    
    train_df = add_basic_features(train_df)
    
    # Handle missing values using TRAIN statistics only
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    train_medians = train_df[numeric_cols].median()
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_medians)
    
    # Label Encoding for categorical variables
    artist_le = LabelEncoder()
    song_le = LabelEncoder()
    country_le = LabelEncoder()
    
    train_df["artist_enc"] = artist_le.fit_transform(train_df["artist"].astype(str))
    train_df["song_enc"] = song_le.fit_transform(train_df["song"].astype(str))
    train_df["country_enc"] = country_le.fit_transform(train_df["country"].astype(str))

    # OneHot Encoding for album type
    album_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    album_enc = album_ohe.fit_transform(train_df[["album_type"]])
    
    album_feature_names = [f"album_{name}" for name in album_ohe.categories_[0]]
    album_df = pd.DataFrame(album_enc, columns=album_feature_names, index=train_df.index)

    # Define feature sets for each target
    position_features = [
        "popularity", "popularity_squared", "popularity_log",
        "duration_ms", "duration_minutes", "duration_log",
        "is_explicit", "total_tracks",
        "days_since_release", "weeks_since_release", "months_since_release",
        "is_weekend", "is_holiday_season", "is_summer",
        "artist_enc", "song_enc", "country_enc"
    ]
    popularity_features = [
        "position", "position_squared", "position_log",
        "duration_ms", "duration_minutes", "duration_log",
        "is_explicit", "total_tracks",
        "days_since_release", "weeks_since_release", "months_since_release",
        "is_weekend", "is_holiday_season", "is_summer",
        "artist_enc", "song_enc", "country_enc"
    ]
    
    # Scale numerical features for each target
    position_scaler = StandardScaler()
    position_scaled = position_scaler.fit_transform(train_df[position_features])
    position_scaled_df = pd.DataFrame(position_scaled, columns=position_features, index=train_df.index)        
    popularity_scaler = StandardScaler()
    popularity_scaled = popularity_scaler.fit_transform(train_df[popularity_features])
    popularity_scaled_df = pd.DataFrame(popularity_scaled, columns=popularity_features, index=train_df.index)

    # Cyclical features (same for both targets)
    cyclical_features = [
        'day_of_week_sin', 'day_of_week_cos', 
        'month_sin', 'month_cos',
        'day_of_month_sin', 'day_of_month_cos'
    ]
    cyclical_df = train_df[cyclical_features]

    # Combine features for each target
    position_train_features = pd.concat([position_scaled_df, album_df, cyclical_df], axis=1)
    popularity_train_features = pd.concat([popularity_scaled_df, album_df, cyclical_df], axis=1)
    
    # Target scaling
    position_target_scaler = MinMaxScaler(feature_range=(0, 1))
    popularity_target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_df["position_norm"] = position_target_scaler.fit_transform(train_df[["position"]])
    train_df["popularity_norm"] = popularity_target_scaler.fit_transform(train_df[["popularity"]])

    # Store preprocessing objects
    preprocessors = {
        'artist_le': artist_le,
        'song_le': song_le, 
        'country_le': country_le,
        'album_ohe': album_ohe,
        'position_scaler': position_scaler,
        'popularity_scaler': popularity_scaler,
        'position_target_scaler': position_target_scaler,
        'popularity_target_scaler': popularity_target_scaler,
        'train_medians': train_medians.to_dict(),
        'position_features': position_features,
        'popularity_features': popularity_features,
        'cyclical_features': cyclical_features,
        'album_feature_names': album_feature_names
    }

    del album_enc, position_scaled, popularity_scaled
    gc.collect()

    return position_train_features, popularity_train_features, train_df, preprocessors

def prepare_features_test_only(test_df, preprocessors):
    """Apply preprocessing to test set using ONLY parameters learned from training set"""
    
    test_df = add_basic_features(test_df)
    
    # Handle missing values using TRAIN statistics only
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in preprocessors['train_medians']:
            test_df[col] = test_df[col].fillna(preprocessors['train_medians'][col])
        else:
            test_df[col] = test_df[col].fillna(0)
    
    # Apply label encoding using fitted encoders from train
    test_df["artist_enc"] = test_df["artist"].astype(str).apply(
        lambda x: preprocessors['artist_le'].transform([x])[0] 
        if x in preprocessors['artist_le'].classes_ else -1
    )
    test_df["song_enc"] = test_df["song"].astype(str).apply(
        lambda x: preprocessors['song_le'].transform([x])[0] 
        if x in preprocessors['song_le'].classes_ else -1
    )
    test_df["country_enc"] = test_df["country"].astype(str).apply(
        lambda x: preprocessors['country_le'].transform([x])[0] 
        if x in preprocessors['country_le'].classes_ else -1
    )

    # Apply OneHot encoding
    album_enc = preprocessors['album_ohe'].transform(test_df[["album_type"]])
    album_df = pd.DataFrame(album_enc, columns=preprocessors['album_feature_names'], index=test_df.index)

    # Apply numerical scaling for both targets
    position_scaled = preprocessors['position_scaler'].transform(test_df[preprocessors['position_features']])
    position_scaled_df = pd.DataFrame(position_scaled, columns=preprocessors['position_features'], index=test_df.index)
    popularity_scaled = preprocessors['popularity_scaler'].transform(test_df[preprocessors['popularity_features']])
    popularity_scaled_df = pd.DataFrame(popularity_scaled, columns=preprocessors['popularity_features'], index=test_df.index)

    # Cyclical features
    cyclical_df = test_df[preprocessors['cyclical_features']]

    # Combine features for each target
    position_test_features = pd.concat([position_scaled_df, album_df, cyclical_df], axis=1)
    popularity_test_features = pd.concat([popularity_scaled_df, album_df, cyclical_df], axis=1)
    
    # Apply target scaling
    test_df["position_norm"] = preprocessors['position_target_scaler'].transform(test_df[["position"]])
    test_df["popularity_norm"] = preprocessors['popularity_target_scaler'].transform(test_df[["popularity"]])

    del album_enc, position_scaled, popularity_scaled
    gc.collect()

    return position_test_features, popularity_test_features, test_df

def tune_hyperparameters(X_train, y_train, target_name, alpha_range, cv_folds=5, random_state=42, n_jobs=-1):
    """Tune Ridge regression hyperparameters using cross-validation for a specific target"""
    
    print(f"Tuning Ridge regression hyperparameters for {target_name}...")
    print(f"Testing alpha values: {alpha_range}")
    
    # Use TimeSeriesSplit for temporal data
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Create Ridge model for grid search
    ridge = Ridge(random_state=random_state)
    
    # Define parameter grid
    param_grid = {'alpha': alpha_range}
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        ridge,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Store results
    best_alpha = grid_search.best_params_['alpha']
    cv_results = grid_search.cv_results_
    
    print(f"Best alpha found for {target_name}: {best_alpha}")
    print(f"Best CV score (neg_MAE): {grid_search.best_score_:.4f}")
    
    # Display top 5 results
    results_df = pd.DataFrame(cv_results)
    results_df = results_df.sort_values('mean_test_score', ascending=False)
    
    print(f"\nTop 5 hyperparameter combinations for {target_name}:")
    print(results_df[['param_alpha', 'mean_test_score', 'std_test_score']].head())
    
    return grid_search.best_estimator_, best_alpha, cv_results

def train_models(X_position_train, y_position_train, X_popularity_train, y_popularity_train, 
                alpha_range, cv_folds=5, random_state=42, n_jobs=-1, tune_hyperparams=True):
    """Train Ridge regression models for both targets"""
    
    if tune_hyperparams:
        print("Training Ridge regression models with hyperparameter tuning...")
        
        # Train position model
        position_model, position_best_alpha, position_cv_results = tune_hyperparameters(
            X_position_train, y_position_train, "position", alpha_range, cv_folds, random_state, n_jobs
        )
        
        # Train popularity model
        popularity_model, popularity_best_alpha, popularity_cv_results = tune_hyperparameters(
            X_popularity_train, y_popularity_train, "popularity", alpha_range, cv_folds, random_state, n_jobs
        )
        
    else:    
        position_model = Ridge(alpha=1.0, random_state=random_state)
        position_model.fit(X_position_train, y_position_train)
        position_best_alpha = 1.0
        position_cv_results = None
        
        popularity_model = Ridge(alpha=1.0, random_state=random_state)
        popularity_model.fit(X_popularity_train, y_popularity_train)
        popularity_best_alpha = 1.0
        popularity_cv_results = None
    
    print("Ridge models training completed!")
    
    return position_model, popularity_model, position_best_alpha, popularity_best_alpha, position_cv_results, popularity_cv_results

def calculate_comprehensive_metrics(y_true, y_pred, include_position_metrics=True):
    """Calculate comprehensive metrics for evaluation"""
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    spearman_corr, _ = spearmanr(y_true.flatten(), y_pred.flatten())
    kendall_corr = kendalltau(y_true.flatten(), y_pred.flatten())[0]
    
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    # Base metrics that are always included
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "spearman": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        "kendall": float(kendall_corr) if not np.isnan(kendall_corr) else 0.0,
    }
    
    # Only include position-specific metrics if requested (for position model)
    if include_position_metrics:
        pred_rounded = np.round(y_pred).astype(int)
        true_rounded = np.round(y_true).astype(int)
        
        exact_accuracy = np.mean(pred_rounded == true_rounded)
        metrics["exact_accuracy"] = float(exact_accuracy)
        
        # Accuracies (only for position)
        accuracies = {}
        for k in [1, 3, 5, 10, 20, 30]:
            accuracies[f"top{k}_accuracy"] = np.mean(np.abs(pred_rounded - true_rounded) <= k)
        
        # Hit rates (only for position)
        hit_rates = {}
        for pos in [10, 20, 30]:
            hit_rates[f"top{pos}_hit_rate"] = np.mean(
                (true_rounded <= pos) & (pred_rounded <= pos)
            )
        
        # Add position-specific metrics
        metrics.update({k: float(v) for k, v in accuracies.items()})
        metrics.update({k: float(v) for k, v in hit_rates.items()})
    
    return metrics

def calculate_country_metrics(X_test, y_test, countries_test, model, target_scaler, target_name):
    """Calculate detailed metrics for each country for a specific target"""
    
    y_pred_norm = model.predict(X_test)
    
    y_pred = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    country_metrics = {}
    unique_countries = np.unique(countries_test)
    
    for country in unique_countries:
        mask = countries_test == country
        
        if np.sum(mask) == 0:
            continue
            
        pred_country = y_pred[mask]
        true_country = y_true[mask]
        
        # Include position metrics only for position model, exclude for popularity model
        include_position_metrics = (target_name == "position")
        metrics = calculate_comprehensive_metrics(
            true_country, pred_country, include_position_metrics=include_position_metrics
        )
        metrics = {"samples": int(np.sum(mask)), **metrics}
        
        country_metrics[country] = metrics
    
    return country_metrics, y_pred, y_true

def save_predictions_to_csv(y_true, y_pred, countries_test, artists_test, songs_test, target_name, output_dir):
    """Save detailed predictions to CSV file for a specific target"""
    
    results_df = pd.DataFrame({
        'country': countries_test,
        'artist': artists_test,
        'song': songs_test,
        f'true_{target_name}': y_true,
        f'predicted_{target_name}': y_pred,
        'prediction_error': np.abs(y_true - y_pred),
        'prediction_error_percentage': np.abs((y_true - y_pred) / np.maximum(y_true, 1)) * 100
    })
    
    results_df[f'predicted_{target_name}_rounded'] = np.round(results_df[f'predicted_{target_name}']).astype(int)
    results_df[f'true_{target_name}_rounded'] = np.round(results_df[f'true_{target_name}']).astype(int)
    
    for k in [1, 3, 5, 10]:
        results_df[f'within_top{k}'] = (np.abs(results_df[f'predicted_{target_name}_rounded'] - results_df[f'true_{target_name}_rounded']) <= k).astype(int)
    
    results_df = results_df.sort_values(['country', 'prediction_error'])
    
    csv_path = os.path.join(output_dir, f'detailed_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Summary statistics
    summary_stats = results_df.groupby('country').agg({
        'prediction_error': ['mean', 'std', 'min', 'max'],
        'prediction_error_percentage': ['mean', 'std'],
        'within_top1': 'mean',
        'within_top3': 'mean',
        'within_top5': 'mean',
        'within_top10': 'mean'
    }).round(4)
    
    summary_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary_stats.columns]
    
    summary_path = os.path.join(output_dir, f'country_prediction_summary.csv')
    summary_stats.to_csv(summary_path)
    
    return results_df, summary_stats

def load_and_process_data(data_path):
    """Load and process data with better error handling"""
    
    dfs = []
    
    files = glob.glob(data_path)
    print(f"Found {len(files)} files to process")
    
    for file in files:
        try:
            df = pd.read_csv(file)
            country = extract_country_name(file)
            df["country"] = country
            df["is_explicit"] = df["is_explicit"].astype(int)
            
            cols_to_drop = ["album_cover_url"]
            df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
            
            dfs.append(df)
            print(f"Loaded {len(df)} records from {country}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError("No data files found or loaded successfully")
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by=["country", "date", "position"]).reset_index(drop=True)
    
    print(f"Total records loaded: {len(df)}")
    print(f"Countries: {sorted(df['country'].unique())}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def save_model_and_results(target_name, model, country_metrics, output_dir):
    """Save model, preprocessors, and results for a specific target"""
    
    # Save model
    joblib.dump(model, f"{output_dir}/ridge_model_{target_name}.joblib")
    
    # Save metrics
    with open(f"{output_dir}/country_metrics.json", "w") as f:
        json.dump(country_metrics, f, indent=2)

def save_preprocessors(preprocessors, position_output_dir, popularity_output_dir, position_best_alpha, popularity_best_alpha, alpha_range, cv_folds):
    """Save preprocessors (shared between both targets)"""
  
    # Add model configuration
    preprocessors['model_config'] = {
        'position_best_alpha': position_best_alpha,
        'popularity_best_alpha': popularity_best_alpha,
        'alpha_range_tested': alpha_range,
        'cv_folds': cv_folds,
        'num_artists': len(preprocessors['artist_le'].classes_),
        'num_songs': len(preprocessors['song_le'].classes_),
        'num_countries': len(preprocessors['country_le'].classes_)
    }
    
    # Save to both directories
    joblib.dump(preprocessors, f"{position_output_dir}/ridge_preprocessors.joblib")
    joblib.dump(preprocessors, f"{popularity_output_dir}/ridge_preprocessors.joblib")

class RidgeMultiTargetRegressor:
    """
    Ridge Regression model for Spotify chart position AND popularity prediction with hyperparameter tuning.
    Prevents data leakage by fitting all preprocessors only on training data.
    Trains separate models for position and popularity prediction.
    """
    
    def __init__(self, 
                 data_path="/TopSpotifAI/data/*.csv",
                 songs_per_day=50,
                 output_dir="/TopSpotifAI/models/RidgeRegressor/out",
                 alpha_range=None,
                 cv_folds=5,
                 random_state=42,
                 n_jobs=-1):
        self.data_path = data_path
        self.songs_per_day = songs_per_day
        self.output_dir = output_dir
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Default alpha range for hyperparameter search
        if alpha_range is None:
            self.alpha_range = [0.01, 0.1, 1, 10, 100]
        else:
            self.alpha_range = alpha_range
        
        # Create output directories for both targets
        self.position_output_dir = os.path.join(self.output_dir, "position")
        self.popularity_output_dir = os.path.join(self.output_dir, "popularity")
        os.makedirs(self.position_output_dir, exist_ok=True)
        os.makedirs(self.popularity_output_dir, exist_ok=True)
        
        # Initialize models and preprocessors for both targets
        self.position_model = None
        self.popularity_model = None
        self.position_best_alpha = None
        self.popularity_best_alpha = None
        self.preprocessors = None
        self.position_cv_results = None
        self.popularity_cv_results = None
    
    def load_existing_models(self):
        """Load existing models and preprocessors if they exist"""
       
        position_model_path = f"{self.position_output_dir}/ridge_model_position.joblib"
        popularity_model_path = f"{self.popularity_output_dir}/ridge_model_popularity.joblib"
        preprocessors_path = f"{self.position_output_dir}/ridge_preprocessors.joblib"
        
        models_exist = (os.path.exists(position_model_path) and 
                       os.path.exists(popularity_model_path) and 
                       os.path.exists(preprocessors_path))
        
        if models_exist:
            print("Loading existing models...")
            try:
                self.position_model = joblib.load(position_model_path)
                self.popularity_model = joblib.load(popularity_model_path)
                self.preprocessors = joblib.load(preprocessors_path)
                
                # Load model configuration if available
                if 'model_config' in self.preprocessors:
                    config = self.preprocessors['model_config']
                    self.position_best_alpha = config.get('position_best_alpha')
                    self.popularity_best_alpha = config.get('popularity_best_alpha')
                
                print("Models loaded successfully!")
                print(f"Position model alpha: {self.position_best_alpha}")
                print(f"Popularity model alpha: {self.popularity_best_alpha}")
                
                return True
            except Exception as e:
                print(f"Error loading models: {e}")
                print("Will train new models...")
                return False
        else:
            print("No existing models found. Training new models...")
            return False
    
    def fit(self, tune_hyperparams=True):
        """Main training method for both targets"""
        
        # Try to load existing models first
        if self.load_existing_models():
            print("Using existing trained models. Skipping training.")
            return {
                'position_model': self.position_model,
                'popularity_model': self.popularity_model,
                'position_metrics': None,  # Not calculated when loading existing models
                'popularity_metrics': None,  # Not calculated when loading existing models
                'position_country_metrics': None,
                'popularity_country_metrics': None
            }
        
        try:
            # Create TensorBoard writers
            position_writer = SummaryWriter(f"/TopSpotifAI/models/RidgeRegressor/runs/position_model")
            popularity_writer = SummaryWriter(f"/TopSpotifAI/models/RidgeRegressor/runs/popularity_model")
        
            # Load and process data
            print("Loading data...")
            df = load_and_process_data(data_path=self.data_path)

            # Split data by country to prevent data leakage
            train_df, test_df = split_data_by_country(df)
            
            # Prepare features (fit preprocessors only on training data)
            X_position_train, X_popularity_train, train_df, preprocessors = prepare_features_train_only(train_df)
            X_position_test, X_popularity_test, test_df = prepare_features_test_only(test_df, preprocessors)
            
            # Extract targets
            y_position_train = train_df["position_norm"].values
            y_popularity_train = train_df["popularity_norm"].values
            y_position_test = test_df["position_norm"].values
            y_popularity_test = test_df["popularity_norm"].values
            
            # Train models
            position_model, popularity_model, position_best_alpha, popularity_best_alpha, _, _ = train_models(
                X_position_train, y_position_train,
                X_popularity_train, y_popularity_train,
                tune_hyperparams=tune_hyperparams,
                alpha_range=self.alpha_range
            )
            
            # Calculate country metrics for POSITION (with accuracies and hit rates)
            print("\n" + "="*50)
            print("EVALUATING POSITION MODEL")
            print("="*50)
            
            position_country_metrics, y_position_pred, y_position_true = calculate_country_metrics(
                X_position_test, y_position_test, test_df["country"].values,
                position_model, preprocessors['position_target_scaler'], "position"
            )
            
            # Calculate overall metrics for position
            position_overall_metrics = calculate_comprehensive_metrics(
                y_position_true, y_position_pred, include_position_metrics=True
            )
            
            # Log only loss metrics to TensorBoard
            for i, (country, metrics) in enumerate(position_country_metrics.items()):
                position_writer.add_scalar(f"MAE/{country}", metrics["mae"], i)
                position_writer.add_scalar(f"RMSE/{country}", metrics["rmse"], i)
            
            for metric, value in position_overall_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Save position predictions to CSV
            save_predictions_to_csv(
                y_position_true, y_position_pred, test_df["country"].values,
                test_df["artist"].values, test_df["song"].values,
                "position", self.position_output_dir
            )
            
            # Save position model and results
            save_model_and_results(
                "position", position_model, position_country_metrics,
                self.position_output_dir
            )
            
            # Calculate country metrics for POPULARITY
            print("\n" + "="*50)
            print("EVALUATING POPULARITY MODEL")
            print("="*50)
            
            popularity_country_metrics, y_popularity_pred, y_popularity_true = calculate_country_metrics(
                X_popularity_test, y_popularity_test, test_df["country"].values,
                popularity_model, preprocessors['popularity_target_scaler'], "popularity"
            )
            
            # Calculate overall metrics for popularity (without position-specific metrics)
            popularity_overall_metrics = calculate_comprehensive_metrics(
                y_popularity_true, y_popularity_pred, include_position_metrics=False
            )
            
            # Log only loss metrics to TensorBoard
            popularity_writer.add_scalar('Loss/MAE', popularity_overall_metrics['mae'], 0)
            popularity_writer.add_scalar('Loss/RMSE', popularity_overall_metrics['rmse'], 0)
            
            for metric, value in popularity_overall_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Save popularity predictions to CSV
            save_predictions_to_csv(
                y_popularity_true, y_popularity_pred, test_df["country"].values,
                test_df["artist"].values, test_df["song"].values,
                "popularity", self.popularity_output_dir
            )
            
            # Save popularity model and results
            save_model_and_results(
                "popularity", popularity_model, popularity_country_metrics, 
                self.popularity_output_dir
            )
            
            # Save shared preprocessors
            save_preprocessors(preprocessors, self.position_output_dir, 
                self.popularity_output_dir, position_best_alpha, 
                popularity_best_alpha, self.alpha_range, self.cv_folds
            )
            
            # Close TensorBoard writers
            position_writer.close()
            popularity_writer.close()
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Position model saved to: {self.position_output_dir}")
            print(f"Popularity model saved to: {self.popularity_output_dir}")
            print(f"Best alpha for position: {position_best_alpha}")
            print(f"Best alpha for popularity: {popularity_best_alpha}")
            print("Run 'tensorboard --logdir=runs/RidgeRegressor' to view the logs")
            
            return {
                'position_model': position_model,
                'popularity_model': popularity_model,
                'position_metrics': position_overall_metrics,
                'popularity_metrics': popularity_overall_metrics,
                'position_country_metrics': position_country_metrics,
                'popularity_country_metrics': popularity_country_metrics
            }
        
        except Exception as e:
            print(f"Error during training: {e}")
            raise
                
def main():
    # Initialize the multi-target regressor
    regressor = RidgeMultiTargetRegressor()
    
    # Train both models with hyperparameter tuning
    regressor.fit(tune_hyperparams=True)
    
    print("\nTraining completed!")
    
if __name__ == "__main__":
    main()