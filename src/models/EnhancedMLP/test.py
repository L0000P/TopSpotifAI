import os
import glob
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau
import torch.nn.functional as F

class EnhancedMLP(nn.Module):
    """Enhanced MLP with improved architecture for multi-target prediction"""
    def __init__(self, num_features, num_artists, num_songs, num_countries=None):
        super().__init__()
        
        # Enhanced embedding dimensions with better initialization
        self.artist_emb = nn.Embedding(num_artists, 128)
        self.song_emb = nn.Embedding(num_songs, 128)
        
        # Add country embedding if available
        self.country_emb = None
        if num_countries:
            self.country_emb = nn.Embedding(num_countries, 64)
            embedding_dim = 128 + 128 + 64
        else:
            embedding_dim = 128 + 128
            
        input_dim = num_features + embedding_dim
        
        # Initialize embeddings with better initialization
        nn.init.normal_(self.artist_emb.weight, mean=0, std=0.1)
        nn.init.normal_(self.song_emb.weight, mean=0, std=0.1)
        if self.country_emb:
            nn.init.normal_(self.country_emb.weight, mean=0, std=0.1)
        
        # Multi-head attention mechanism for feature importance
        self.attention_heads = 8  # Common divisor
        attention_dim = ((input_dim // self.attention_heads) * self.attention_heads)
        
        # Project to attention dimension
        self.input_projection = nn.Linear(input_dim, attention_dim) if input_dim != attention_dim else nn.Identity()
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim, 
            num_heads=self.attention_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Project back to original dimension
        self.output_projection = nn.Linear(attention_dim, input_dim) if input_dim != attention_dim else nn.Identity()
        
        # Residual blocks for better gradient flow
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(input_dim, 1024),
            self._make_residual_block(1024, 512),
            self._make_residual_block(512, 256)
        ])
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.position_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Popularity head
        self.popularity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def _make_residual_block(self, in_dim, out_dim):
        """Create a residual block with skip connection"""
        return nn.ModuleDict({
            'linear1': nn.Linear(in_dim, out_dim),
            'bn1': nn.BatchNorm1d(out_dim),
            'linear2': nn.Linear(out_dim, out_dim),
            'bn2': nn.BatchNorm1d(out_dim),
            'skip': nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity(),
            'dropout': nn.Dropout(0.3)
        })
    
    def _residual_forward(self, x, block):
        """Forward pass through residual block"""
        identity = block['skip'](x)
        
        out = F.gelu(block['bn1'](block['linear1'](x)))
        out = block['dropout'](out)
        out = block['bn2'](block['linear2'](out))
        
        out = out + identity
        return F.gelu(out)
        
    def forward(self, x_num, x_artist, x_song, x_country=None):
        """Enhanced forward pass with attention and residual connections"""
        
        # Get embeddings
        artist_emb = self.artist_emb(x_artist)
        song_emb = self.song_emb(x_song)
        
        # Combine features
        if x_country is not None and self.country_emb is not None:
            country_emb = self.country_emb(x_country)
            x = torch.cat([x_num, artist_emb, song_emb, country_emb], dim=1)
        else:
            x = torch.cat([x_num, artist_emb, song_emb], dim=1)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Project to attention dimension
        x_projected = self.input_projection(x)
        
        # Self-attention mechanism
        x_unsqueezed = x_projected.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attn_out = attn_out.squeeze(1)  # Remove sequence dimension
        
        # Project back and add residual connection
        x = x + self.output_projection(attn_out)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = self._residual_forward(x, block)
        
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Task-specific predictions
        position_pred = self.position_head(shared_features)
        popularity_pred = self.popularity_head(shared_features)
        
        return position_pred, popularity_pred

def add_advanced_features(df):
    """Add more sophisticated features"""
    
    # Existing datetime features
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["year"] = df["date"].dt.year
    
    # Cyclical encoding
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
    
    # Advanced features
    df["release_year"] = pd.to_datetime(df["release_date"]).dt.year
    df["days_since_release"] = (df["date"] - pd.to_datetime(df["release_date"])).dt.days
    df["weeks_since_release"] = df["days_since_release"] / 7
    df["months_since_release"] = df["days_since_release"] / 30.44
    
    # Popularity-based features
    df["popularity_squared"] = df["popularity"] ** 2
    df["popularity_log"] = np.log1p(df["popularity"])
    
    # Duration-based features
    df["duration_minutes"] = df["duration_ms"] / 60000
    df["duration_log"] = np.log1p(df["duration_ms"])
    
    # Categorical features
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday_season"] = ((df["month"] == 12) | (df["month"] == 1)).astype(int)
    df["is_summer"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)
    
    return df

def create_lag_features(df, country_col="country", date_col="date", target_col="position"):
    """Create lag features for time series patterns"""
    df = df.sort_values([country_col, date_col]).copy()
    
    # Create lag features by country
    for country in df[country_col].unique():
        mask = df[country_col] == country
        country_data = df[mask].copy()
        
        # 1-day, 3-day, 7-day lags
        for lag in [1, 3, 7]:
            lag_col = f"{target_col}_lag_{lag}"
            df.loc[mask, lag_col] = country_data[target_col].shift(lag)
        
        # Rolling averages
        for window in [3, 7, 14]:
            rolling_col = f"{target_col}_rolling_{window}"
            df.loc[mask, rolling_col] = country_data[target_col].rolling(window=window, min_periods=1).mean()
    
    # Fill NaN values with median
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    for col in lag_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def prepare_enhanced_features(df, config):
    """Enhanced feature preparation with configuration parameter"""
    
    # Create lag features for both targets
    df = create_lag_features(df, target_col="position")
    df = create_lag_features(df, target_col="popularity")
    
    # Label Encoding with unknown handling
    artist_le = LabelEncoder()
    song_le = LabelEncoder()
    country_le = LabelEncoder()
    df["artist_enc"] = artist_le.fit_transform(df["artist"])
    df["song_enc"] = song_le.fit_transform(df["song"])
    df["country_enc"] = country_le.fit_transform(df["country"])

    # Advanced feature engineering
    df = add_advanced_features(df)

    # OneHot Encoding for categorical features
    album_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    album_enc = album_ohe.fit_transform(df[["album_type"]])

    # Robust numerical feature scaling (excluding popularity from features)
    numerical_features = [
        "duration_ms", "duration_minutes", "duration_log",
        "is_explicit", "total_tracks",
        "days_since_release", "weeks_since_release", "months_since_release",
        "is_weekend", "is_holiday_season", "is_summer"
    ]
    
    # Add lag features to numerical features (including both position and popularity lags)
    lag_features = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    numerical_features.extend(lag_features)
    
    # Handle missing values
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
    
    # Scale numerical features
    scaler_num = StandardScaler()
    num_feats = scaler_num.fit_transform(df[numerical_features])

    # Cyclical features
    cyclical_feats = df[[
        'day_of_week_sin', 'day_of_week_cos', 
        'month_sin', 'month_cos',
        'day_of_month_sin', 'day_of_month_cos'
    ]].values

    # Combine features
    X_num_all = np.hstack([num_feats, album_enc, cyclical_feats])

    # Scale both targets
    pos_scaler = MinMaxScaler(feature_range=(0, 1))
    pop_scaler = MinMaxScaler(feature_range=(0, 1))
    df["position_norm"] = pos_scaler.fit_transform(df[["position"]])
    df["popularity_norm"] = pop_scaler.fit_transform(df[["popularity"]])

    # Prepare data containers
    train_features, test_features = [], []
    train_artist, test_artist = [], []
    train_song, test_song = [], []
    train_country, test_country = [], []
    train_pos_target, test_pos_target = [], []
    train_pop_target, test_pop_target = [], []

    # Process each country with stratified split
    for country in tqdm(df["country"].unique(), desc="Processing countries"):
        cdf = df[df["country"] == country].sort_values("date").copy()
        
        # Select top songs per day
        cdf_daily = cdf.sort_values("position").groupby("date").head(config['songs_per_day']).reset_index(drop=True)
        all_dates = sorted(cdf_daily["date"].unique())

        # Temporal split with validation consideration
        n_dates = len(all_dates)
        train_split = int(n_dates * 0.8)
        
        train_dates = all_dates[:train_split]
        test_dates = all_dates[train_split:]

        train_df = cdf_daily[cdf_daily["date"].isin(train_dates)]
        test_df = cdf_daily[cdf_daily["date"].isin(test_dates)]

        # Collect features
        train_features.append(X_num_all[train_df.index])
        test_features.append(X_num_all[test_df.index])

        train_artist.append(train_df["artist_enc"].values)
        test_artist.append(test_df["artist_enc"].values)

        train_song.append(train_df["song_enc"].values)
        test_song.append(test_df["song_enc"].values)
        
        train_country.append(train_df["country_enc"].values)
        test_country.append(test_df["country_enc"].values)

        # Both targets
        train_pos_target.append(train_df["position_norm"].values)
        test_pos_target.append(test_df["position_norm"].values)
        train_pop_target.append(train_df["popularity_norm"].values)
        test_pop_target.append(test_df["popularity_norm"].values)

    # Convert to tensors
    X_train_num = torch.tensor(np.vstack(train_features), dtype=torch.float32)
    X_test_num = torch.tensor(np.vstack(test_features), dtype=torch.float32)
    artist_train = torch.tensor(np.hstack(train_artist), dtype=torch.long)
    artist_test = torch.tensor(np.hstack(test_artist), dtype=torch.long)
    song_train = torch.tensor(np.hstack(train_song), dtype=torch.long)
    song_test = torch.tensor(np.hstack(test_song), dtype=torch.long)
    country_train = torch.tensor(np.hstack(train_country), dtype=torch.long)
    country_test = torch.tensor(np.hstack(test_country), dtype=torch.long)
    
    # Both targets as tensors
    y_pos_train = torch.tensor(np.hstack(train_pos_target), dtype=torch.float32).view(-1,1)
    y_pos_test = torch.tensor(np.hstack(test_pos_target), dtype=torch.float32).view(-1,1)
    y_pop_train = torch.tensor(np.hstack(train_pop_target), dtype=torch.float32).view(-1,1)
    y_pop_test = torch.tensor(np.hstack(test_pop_target), dtype=torch.float32).view(-1,1)

    return (
        X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
        X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test,
        artist_le, song_le, country_le, album_ohe, scaler_num, pos_scaler, pop_scaler
    )

def get_enhanced_batches(X_num, X_artist, X_song, X_country, y_pos, y_pop, batch_size, shuffle=True):
    """Enhanced batch generation with optional shuffling for multi-target"""
    
    n = len(y_pos)
    if shuffle:
        indices = torch.randperm(n)
    else:
        indices = torch.arange(n)
        
    for i in range(0, n, batch_size):
        idx = indices[i:i+batch_size]
        yield X_num[idx], X_artist[idx], X_song[idx], X_country[idx], y_pos[idx], y_pop[idx]

def load_and_process_data(data_path):
    """Enhanced data loading with configurable path"""
    
    dfs = []
    for file in glob.glob(data_path):
        try:
            df = pd.read_csv(file)
            country = os.path.splitext(os.path.basename(file))[0]
            df["country"] = country
            df["is_explicit"] = df["is_explicit"].astype(int)
            
            # Drop unnecessary columns if they exist
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
    print(f"Countries: {df['country'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def enhanced_train(model, device, criterion, optimizer, data, scalers, config, run_name="enhanced_mlp", 
                  load_existing=True, model_path=None, preprocessor_path=None):
    """Enhanced training with multi-target support and automatic model loading"""
    
    pos_scaler, pop_scaler = scalers
    
    # Default paths if not provided
    if model_path is None:
        model_path = f"{config['output_dir']}/enhanced_model.pth"
    if preprocessor_path is None:
        preprocessor_path = f"{config['output_dir']}/enhanced_preprocessors.pth"
    
    # Check if we should load existing model
    if load_existing and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully!")
            
            # Return validation predictions for consistency
            (X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
             X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test) = data
            
            X_test_num = X_test_num.to(device)
            artist_test = artist_test.to(device)
            song_test = song_test.to(device)
            country_test = country_test.to(device)
            y_pos_test = y_pos_test.to(device)
            y_pop_test = y_pop_test.to(device)
            
            model.eval()
            with torch.no_grad():
                pos_pred, pop_pred = model(X_test_num, artist_test, song_test, country_test)
            
            print("Skipping training - using loaded model for predictions")
            return model, (pos_pred, pop_pred), (y_pos_test, y_pop_test), [{"epoch": 0, "status": "loaded_existing"}]
        
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Proceeding with training...")
            
    elif load_existing:
        print(f"Model file not found at {model_path}")
        print("Proceeding with training...")
    
    # Proceed with training if not loading existing model
    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], run_name))
    
    (X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
     X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test) = data

    # Move to device
    X_train_num = X_train_num.to(device)
    artist_train = artist_train.to(device)
    song_train = song_train.to(device)
    country_train = country_train.to(device)
    y_pos_train = y_pos_train.to(device)
    y_pop_train = y_pop_train.to(device)
    X_test_num = X_test_num.to(device)
    artist_test = artist_test.to(device)
    song_test = song_test.to(device)
    country_test = country_test.to(device)
    y_pos_test = y_pos_test.to(device)
    y_pop_test = y_pop_test.to(device)

    # Enhanced learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-7
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    
    # Best model tracking
    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0
    metrics_log = []
    
    # Training loop
    for epoch in tqdm(range(1, config['epochs']+1), desc="Training"):
        model.train()
        train_losses = []
        train_pos_losses = []
        train_pop_losses = []
        
        # Training phase
        for xb_num, xb_artist, xb_song, xb_country, yb_pos, yb_pop in get_enhanced_batches(
            X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train, config['batch_size']
        ):
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda", enabled=device.type == 'cuda'):
                pos_output, pop_output = model(xb_num, xb_artist, xb_song, xb_country)
                pos_loss = criterion(pos_output, yb_pos)
                pop_loss = criterion(pop_output, yb_pop)
                # Combined loss with equal weighting (you can adjust weights)
                total_loss = pos_loss + pop_loss
            
            if device.type == 'cuda':
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            train_losses.append(total_loss.item())
            train_pos_losses.append(pos_loss.item())
            train_pop_losses.append(pop_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_train_pos_loss = np.mean(train_pos_losses)
        avg_train_pop_loss = np.mean(train_pop_losses)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            pos_pred, pop_pred = model(X_test_num, artist_test, song_test, country_test)
            pos_val_loss = criterion(pos_pred, y_pos_test).item()
            pop_val_loss = criterion(pop_pred, y_pop_test).item()
            val_loss = pos_val_loss + pop_val_loss
            
            # Calculate comprehensive metrics for position
            pos_pred_denorm = pos_scaler.inverse_transform(pos_pred.cpu().numpy())
            y_pos_test_denorm = pos_scaler.inverse_transform(y_pos_test.cpu().numpy())
            
            pos_mae = mean_absolute_error(y_pos_test_denorm, pos_pred_denorm)
            pos_mse = mean_squared_error(y_pos_test_denorm, pos_pred_denorm)
            pos_rmse = np.sqrt(pos_mse)
            
            pos_spearman_corr, _ = spearmanr(y_pos_test_denorm.flatten(), pos_pred_denorm.flatten())
            pos_kendall_corr = kendalltau(y_pos_test_denorm.flatten(), pos_pred_denorm.flatten())[0]
            
            # Calculate comprehensive metrics for popularity
            pop_pred_denorm = pop_scaler.inverse_transform(pop_pred.cpu().numpy())
            y_pop_test_denorm = pop_scaler.inverse_transform(y_pop_test.cpu().numpy())
            
            pop_mae = mean_absolute_error(y_pop_test_denorm, pop_pred_denorm)
            pop_mse = mean_squared_error(y_pop_test_denorm, pop_pred_denorm)
            pop_rmse = np.sqrt(pop_mse)
            
            pop_spearman_corr, _ = spearmanr(y_pop_test_denorm.flatten(), pop_pred_denorm.flatten())
            pop_kendall_corr = kendalltau(y_pop_test_denorm.flatten(), pop_pred_denorm.flatten())[0]
            
            # Top-k accuracies for position
            pos_pred_rounded = np.round(pos_pred_denorm).astype(int)
            pos_true_rounded = np.round(y_pos_test_denorm).astype(int)
            
            pos_accuracies = {}
            for k in [1, 3, 5, 10, 20, 30]:
                pos_accuracies[f"pos_top{k}"] = np.mean(np.abs(pos_pred_rounded - pos_true_rounded) <= k)

        # Log metrics TensorBoard
        writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
        writer.add_scalar("Loss/Train_Position", avg_train_pos_loss, epoch)
        writer.add_scalar("Loss/Train_Popularity", avg_train_pop_loss, epoch)
        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Val_Position", pos_val_loss, epoch)
        writer.add_scalar("Loss/Val_Popularity", pop_val_loss, epoch)
        
        # Position metrics TensorBoard
        writer.add_scalar("Position/MAE", pos_mae, epoch)
        writer.add_scalar("Position/RMSE", pos_rmse, epoch)
        writer.add_scalar("Position/Spearman", pos_spearman_corr, epoch)
        writer.add_scalar("Position/Kendall", pos_kendall_corr, epoch)
        
        # Popularity metrics TensorBoard
        writer.add_scalar("Popularity/MAE", pop_mae, epoch)
        writer.add_scalar("Popularity/RMSE", pop_rmse, epoch)
        writer.add_scalar("Popularity/Spearman", pop_spearman_corr, epoch)
        writer.add_scalar("Popularity/Kendall", pop_kendall_corr, epoch)
        
        for k, acc in pos_accuracies.items():
            writer.add_scalar(f"Position_Accuracy/{k}", acc, epoch)
        
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss_total": float(avg_train_loss),
            "train_loss_position": float(avg_train_pos_loss),
            "train_loss_popularity": float(avg_train_pop_loss),
            "val_loss_total": float(val_loss),
            "val_loss_position": float(pos_val_loss),
            "val_loss_popularity": float(pop_val_loss),
            "pos_mae": float(pos_mae),
            "pos_rmse": float(pos_rmse),
            "pos_spearman": float(pos_spearman_corr),
            "pos_kendall": float(pos_kendall_corr),
            "pop_mae": float(pop_mae),
            "pop_rmse": float(pop_rmse),
            "pop_spearman": float(pop_spearman_corr),
            "pop_kendall": float(pop_kendall_corr),
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        epoch_metrics.update({f"{k}_accuracy": float(v) for k, v in pos_accuracies.items()})
        metrics_log.append(epoch_metrics)

        # Progress reporting
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch}/{config['epochs']}")
            print(f"Train Loss: {avg_train_loss:.5f} (Pos: {avg_train_pos_loss:.5f}, Pop: {avg_train_pop_loss:.5f})")
            print(f"Val Loss: {val_loss:.5f} (Pos: {pos_val_loss:.5f}, Pop: {pop_val_loss:.5f})")
            print(f"Position - MAE: {pos_mae:.3f} | RMSE: {pos_rmse:.3f} | Spearman: {pos_spearman_corr:.4f}")
            print(f"Popularity - MAE: {pop_mae:.3f} | RMSE: {pop_rmse:.3f} | Spearman: {pop_spearman_corr:.4f}")
            print(f"Position Top-1 Acc: {pos_accuracies['pos_top1']:.4f} | Top-5 Acc: {pos_accuracies['pos_top5']:.4f} | Top-10 Acc: {pos_accuracies['pos_top10']:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping based on total validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            
            # Save best model immediately
            torch.save(best_model_state, model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.5f}")

    writer.close()
    return model, (pos_pred, pop_pred), (y_pos_test, y_pop_test), metrics_log
    
def calculate_country_metrics_multi_target(model, device, data, scalers, country_le):
    """Calculate detailed metrics for each country for both position and popularity"""
    
    (X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
     X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test) = data
    
    pos_scaler, pop_scaler = scalers
    
    # Move test data to device
    X_test_num = X_test_num.to(device)
    artist_test = artist_test.to(device)
    song_test = song_test.to(device)
    country_test = country_test.to(device)
    y_pos_test = y_pos_test.to(device)
    y_pop_test = y_pop_test.to(device)
    
    model.eval()
    position_metrics = {}
    popularity_metrics = {}
    
    with torch.no_grad():
        
        # Get all predictions
        pos_pred, pop_pred = model(X_test_num, artist_test, song_test, country_test)
        
        # Denormalize predictions and targets
        pos_pred_denorm = pos_scaler.inverse_transform(pos_pred.cpu().numpy())
        y_pos_test_denorm = pos_scaler.inverse_transform(y_pos_test.cpu().numpy())
        pop_pred_denorm = pop_scaler.inverse_transform(pop_pred.cpu().numpy())
        y_pop_test_denorm = pop_scaler.inverse_transform(y_pop_test.cpu().numpy())
        
        # Convert country indices back to names
        country_names = country_test.cpu().numpy()
        
        # Calculate metrics for each country
        for country_idx in np.unique(country_names):
            country_name = country_le.inverse_transform([country_idx])[0]
            mask = country_names == country_idx
            
            if np.sum(mask) == 0:
                continue
            
            # Position metrics
            pos_pred_country = pos_pred_denorm[mask]
            pos_true_country = y_pos_test_denorm[mask]
            
            pos_mae = mean_absolute_error(pos_true_country, pos_pred_country)
            pos_mse = mean_squared_error(pos_true_country, pos_pred_country)
            pos_rmse = np.sqrt(pos_mse)
            
            pos_spearman_corr, _ = spearmanr(pos_true_country.flatten(), pos_pred_country.flatten())
            pos_kendall_corr = kendalltau(pos_true_country.flatten(), pos_pred_country.flatten())[0]
            
            # Top-k accuracies for position
            pos_pred_rounded = np.round(pos_pred_country).astype(int)
            pos_true_rounded = np.round(pos_true_country).astype(int)
            
            pos_accuracies = {}
            for k in [1, 3, 5, 10, 20, 30]:
                pos_accuracies[f"top{k}_accuracy"] = np.mean(np.abs(pos_pred_rounded - pos_true_rounded) <= k)
            
            # Position MAPE
            pos_mape = np.mean(np.abs((pos_true_country - pos_pred_country) / np.maximum(pos_true_country, 1))) * 100
            pos_exact_accuracy = np.mean(pos_pred_rounded == pos_true_rounded)
            
            position_metrics[country_name] = {
                "samples": int(np.sum(mask)),
                "mae": float(pos_mae),
                "rmse": float(pos_rmse),
                "mape": float(pos_mape),
                "spearman": float(pos_spearman_corr) if not np.isnan(pos_spearman_corr) else 0.0,
                "kendall": float(pos_kendall_corr) if not np.isnan(pos_kendall_corr) else 0.0,
                "exact_accuracy": float(pos_exact_accuracy),
                **{k: float(v) for k, v in pos_accuracies.items()}
            }
            
            # Popularity metrics
            pop_pred_country = pop_pred_denorm[mask]
            pop_true_country = y_pop_test_denorm[mask]
            
            pop_mae = mean_absolute_error(pop_true_country, pop_pred_country)
            pop_mse = mean_squared_error(pop_true_country, pop_pred_country)
            pop_rmse = np.sqrt(pop_mse)
            
            pop_spearman_corr, _ = spearmanr(pop_true_country.flatten(), pop_pred_country.flatten())
            pop_kendall_corr = kendalltau(pop_true_country.flatten(), pop_pred_country.flatten())[0]
            
            # Popularity MAPE
            pop_mape = np.mean(np.abs((pop_true_country - pop_pred_country) / np.maximum(pop_true_country, 1))) * 100
            pop_exact_accuracy = np.mean(np.round(pop_pred_country) == np.round(pop_true_country))
            
            popularity_metrics[country_name] = {
                "samples": int(np.sum(mask)),
                "mae": float(pop_mae),
                "rmse": float(pop_rmse),
                "mape": float(pop_mape),
                "spearman": float(pop_spearman_corr) if not np.isnan(pop_spearman_corr) else 0.0,
                "kendall": float(pop_kendall_corr) if not np.isnan(pop_kendall_corr) else 0.0,
                "exact_accuracy": float(pop_exact_accuracy)
            }
    
    return position_metrics, popularity_metrics

def save_separate_predictions(model, device, data, scalers, encoders, config):
    """Save separate predictions for position and popularity targets"""
    
    (X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
     X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test) = data
    
    pos_scaler, pop_scaler = scalers
    artist_le, song_le, country_le = encoders
    
    # Move test data to device
    X_test_num = X_test_num.to(device)
    artist_test = artist_test.to(device)
    song_test = song_test.to(device)
    country_test = country_test.to(device)
    y_pos_test = y_pos_test.to(device)
    y_pop_test = y_pop_test.to(device)
    
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        pos_pred, pop_pred = model(X_test_num, artist_test, song_test, country_test)
        
        # Denormalize
        pos_pred_denorm = pos_scaler.inverse_transform(pos_pred.cpu().numpy())
        y_pos_test_denorm = pos_scaler.inverse_transform(y_pos_test.cpu().numpy())
        pop_pred_denorm = pop_scaler.inverse_transform(pop_pred.cpu().numpy())
        y_pop_test_denorm = pop_scaler.inverse_transform(y_pop_test.cpu().numpy())
        
        # Convert back to original values
        countries = country_le.inverse_transform(country_test.cpu().numpy())
        artists = artist_le.inverse_transform(artist_test.cpu().numpy())
        songs = song_le.inverse_transform(song_test.cpu().numpy())
        
        # Create common columns
        common_data = {
            'country': countries,
            'artist': artists,
            'song': songs
        }
        
        # Position predictions DataFrame
        position_df = pd.DataFrame({
            **common_data,
            'true_position': y_pos_test_denorm.flatten(),
            'predicted_position': pos_pred_denorm.flatten(),
            'prediction_error': np.abs(y_pos_test_denorm.flatten() - pos_pred_denorm.flatten()),
            'prediction_error_percentage': np.abs((y_pos_test_denorm.flatten() - pos_pred_denorm.flatten()) / np.maximum(y_pos_test_denorm.flatten(), 1)) * 100
        })
        
        # Add rounded predictions for position
        position_df['predicted_position_rounded'] = np.round(position_df['predicted_position']).astype(int)
        position_df['true_position_rounded'] = np.round(position_df['true_position']).astype(int)
        
        # Add accuracy flags for position
        for k in [1, 3, 5, 10]:
            position_df[f'within_top{k}'] = (np.abs(position_df['predicted_position_rounded'] - position_df['true_position_rounded']) <= k).astype(int)
        
        # Popularity predictions DataFrame
        popularity_df = pd.DataFrame({
            **common_data,
            'true_popularity': y_pop_test_denorm.flatten(),
            'predicted_popularity': pop_pred_denorm.flatten(),
            'prediction_error': np.abs(y_pop_test_denorm.flatten() - pop_pred_denorm.flatten()),
            'prediction_error_percentage': np.abs((y_pop_test_denorm.flatten() - pop_pred_denorm.flatten()) / np.maximum(y_pop_test_denorm.flatten(), 1)) * 100
        })
        
        # Add rounded predictions for popularity
        popularity_df['predicted_popularity_rounded'] = np.round(popularity_df['predicted_popularity']).astype(int)
        popularity_df['true_popularity_rounded'] = np.round(popularity_df['true_popularity']).astype(int)
        
        # Sort DataFrames
        position_df = position_df.sort_values(['country', 'prediction_error'])
        popularity_df = popularity_df.sort_values(['country', 'prediction_error'])
        
        # Save position predictions
        pos_csv_path = os.path.join(config['output_dir'], 'position', 'position_predictions.csv')
        position_df.to_csv(pos_csv_path, index=False)
        print(f"Position predictions saved to {pos_csv_path}")
        
        # Save popularity predictions 
        pop_csv_path = os.path.join(config['output_dir'], 'popularity', 'popularity_predictions.csv')
        popularity_df.to_csv(pop_csv_path, index=False)
        print(f"Popularity predictions saved to {pop_csv_path}")
         
        return position_df, popularity_df
       
def create_config():
    """Create configuration dictionary"""
    
    return {
        'data_path': "/TopSpotifAI/data/*.csv",
        'log_dir': "/TopSpotifAI/models/EnhancedMLP/runs",
        'output_dir': "/TopSpotifAI/models/EnhancedMLP/out",
        'epochs': 5000,
        'lr': 8e-5,
        'batch_size': 1024,
        'songs_per_day': 50,
        'patience': 500
    }

def main():
    """Enhanced main function for multi-target prediction"""
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    try:
        # Create output directories
        config = create_config()
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'position'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'popularity'), exist_ok=True)
        
        # Load and process data
        print("Loading data...")
        df = load_and_process_data(create_config()['data_path'])
        
        print("Preparing features...")
        (X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
         X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test,
         artist_le, song_le, country_le, album_ohe, scaler_num, pos_scaler, pop_scaler) = prepare_enhanced_features(df, config)

        print(f"Training data shape: {X_train_num.shape}")
        print(f"Test data shape: {X_test_num.shape}")
        print(f"Number of artists: {len(artist_le.classes_)}")
        print(f"Number of songs: {len(song_le.classes_)}")
        print(f"Number of countries: {len(country_le.classes_)}")

        # Initialize enhanced model
        model = EnhancedMLP(
            num_features=X_train_num.shape[1],
            num_artists=len(artist_le.classes_),
            num_songs=len(song_le.classes_),
            num_countries=len(country_le.classes_)
        ).to(device)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Enhanced optimizer and loss
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Train model
        print("Starting training...")
        model_trained, val_preds, y_tests, metrics = enhanced_train(
            model, device, criterion, optimizer,
            (X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
             X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test),
            (pos_scaler, pop_scaler), config
        )

        # Save model and other artifacts
        print("\n" + "="*60)
        print("SAVING MODEL AND ARTIFACTS")
        print("="*60)
        
        print("Saving model state...")
        torch.save(model_trained.state_dict(), f"{config['output_dir']}/enhanced_model.pth")
        
        print("Saving preprocessors...")
        torch.save({
            'artist_le': artist_le,
            'song_le': song_le,
            'country_le': country_le,
            'album_ohe': album_ohe,
            'scaler_num': scaler_num,
            'pos_scaler': pos_scaler,
            'pop_scaler': pop_scaler,
            'model_config': {
                'num_features': X_train_num.shape[1],
                'num_artists': len(artist_le.classes_),
                'num_songs': len(song_le.classes_),
                'num_countries': len(country_le.classes_)
            }
        }, f"{config['output_dir']}/enhanced_preprocessors.pth")

        # Save training metrics
        print("Saving training metrics...")
        with open(f"{config['output_dir']}/enhanced_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save Country Metrics
        print("Calculating country-specific metrics...")
        position_metrics, popularity_metrics = calculate_country_metrics_multi_target(
            model_trained, device, 
            (X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
             X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test),
            (pos_scaler, pop_scaler), country_le
        )
        
        # Save position metrics by country
        print("Saving position metrics by country...")
        position_metrics_df = pd.DataFrame.from_dict(position_metrics, orient='index')
        position_metrics_df.to_csv(f"{config['output_dir']}/position/position_metrics_by_country.csv")
        
        with open(f"{config['output_dir']}/position/position_metrics_by_country.json", "w") as f:
            json.dump(position_metrics, f, indent=2)
        
        # Save popularity metrics by country
        print("Saving popularity metrics by country...")
        popularity_metrics_df = pd.DataFrame.from_dict(popularity_metrics, orient='index')
        popularity_metrics_df.to_csv(f"{config['output_dir']}/popularity/popularity_metrics_by_country.csv")
        
        with open(f"{config['output_dir']}/popularity/popularity_metrics_by_country.json", "w") as f:
            json.dump(popularity_metrics, f, indent=2)

        # Save detailed predictions
        print("Saving detailed predictions...")
        save_separate_predictions(
            model_trained, device,
            (X_train_num, artist_train, song_train, country_train, y_pos_train, y_pop_train,
             X_test_num, artist_test, song_test, country_test, y_pos_test, y_pop_test),
            (pos_scaler, pop_scaler), (artist_le, song_le, country_le), config
        )

        # Print summary
        print("\n" + "="*60)
        print("POSITION METRICS SUMMARY BY COUNTRY")
        print("="*60)
        print(position_metrics_df.round(4))
        
        print("\n" + "="*60)
        print("POPULARITY METRICS SUMMARY BY COUNTRY") 
        print("="*60)
        print(popularity_metrics_df.round(4))

        print(f"\nAll metrics and predictions saved to: {config['output_dir']}")

    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()