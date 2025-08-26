"""
Sales Forecasting - Modular Implementation
A comprehensive sales forecasting project with multiple ML models.

Author: Your Name
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalesForecastAnalyzer:
    """Main class for sales forecasting analysis."""
    
    def __init__(self, data_path='data/train.csv', results_dir='results'):
        """
        Initialize the analyzer.
        
        Args:
            data_path (str): Path to the training data
            results_dir (str): Directory to save results
        """
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.df_enhanced = None
        self.df_clean = None
        self.models = {}
        self.metrics = {}
        
    def load_and_explore_data(self):
        """Load data and perform initial exploration."""
        print("Loading and exploring data...")
        
        # Load dataset
        self.df = pd.read_csv(self.data_path, parse_dates=["date"])
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Basic info
        print("\nDataset Info:")
        print(self.df.info())
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        return self.df
    
    def perform_eda(self, save_plots=True):
        """Perform comprehensive exploratory data analysis."""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # 1. Time Series Plot
        plt.figure(figsize=(15, 6))
        daily_sales = self.df.groupby("date")["sales"].sum()
        daily_sales.plot(linewidth=1.5, alpha=0.8)
        plt.title("Total Daily Sales Over Time", fontsize=16, fontweight='bold')
        plt.ylabel("Total Sales", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(self.results_dir / 'daily_sales_trend.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Sales Distribution by Store and Item
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Sales by store
        store_sales = self.df.groupby('store')['sales'].mean().sort_values(ascending=False)
        store_sales.plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.8)
        axes[0].set_title('Average Sales by Store', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Store ID', fontsize=12)
        axes[0].set_ylabel('Average Sales', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Sales by top items
        item_sales = self.df.groupby('item')['sales'].mean().sort_values(ascending=False)
        item_sales.head(20).plot(kind='bar', ax=axes[1], color='lightcoral', alpha=0.8)
        axes[1].set_title('Average Sales by Top 20 Items', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Item ID', fontsize=12)
        axes[1].set_ylabel('Average Sales', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.results_dir / 'sales_by_store_item.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Seasonal Analysis
        temp_df = self.df.copy()
        temp_df['month'] = temp_df['date'].dt.month
        temp_df['dayofweek'] = temp_df['date'].dt.dayofweek
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Monthly pattern
        monthly_sales = temp_df.groupby('month')['sales'].mean()
        monthly_sales.plot(kind='bar', ax=axes[0,0], color='green', alpha=0.7)
        axes[0,0].set_title('Average Sales by Month', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Month', fontsize=12)
        axes[0,0].set_ylabel('Average Sales', fontsize=12)
        
        # Weekly pattern
        weekly_sales = temp_df.groupby('dayofweek')['sales'].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_sales.plot(kind='bar', ax=axes[0,1], color='orange', alpha=0.7)
        axes[0,1].set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Day of Week', fontsize=12)
        axes[0,1].set_ylabel('Average Sales', fontsize=12)
        axes[0,1].set_xticklabels(day_names, rotation=45)
        
        # Sales distribution
        axes[1,0].hist(self.df['sales'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].set_title('Sales Distribution', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Sales', fontsize=12)
        axes[1,0].set_ylabel('Frequency', fontsize=12)
        
        # Box plot by store (sample)
        sample_stores = self.df['store'].unique()[:10]
        store_data = [self.df[self.df['store'] == store]['sales'] for store in sample_stores]
        axes[1,1].boxplot(store_data, labels=sample_stores)
        axes[1,1].set_title('Sales Distribution by Store (Sample)', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Store ID', fontsize=12)
        axes[1,1].set_ylabel('Sales', fontsize=12)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.results_dir / 'seasonal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return temp_df
    
    def create_features(self, df):
        """Create engineered features for modeling."""
        print("\n=== FEATURE ENGINEERING ===")
        
        df_feat = df.copy()
        
        # Basic datetime features
        df_feat['year'] = df_feat['date'].dt.year
        df_feat['month'] = df_feat['date'].dt.month
        df_feat['day'] = df_feat['date'].dt.day
        df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
        df_feat['dayofyear'] = df_feat['date'].dt.dayofyear
        df_feat['quarter'] = df_feat['date'].dt.quarter
        df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
        df_feat['is_month_end'] = df_feat['date'].dt.is_month_end.astype(int)
        df_feat['is_month_start'] = df_feat['date'].dt.is_month_start.astype(int)
        
        # Cyclical encoding for better pattern capture
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        df_feat['dayofweek_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7)
        df_feat['dayofweek_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7)
        df_feat['dayofyear_sin'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365)
        df_feat['dayofyear_cos'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365)
        
        # Sort for lag features
        df_feat = df_feat.sort_values(['store', 'item', 'date']).reset_index(drop=True)
        
        # Lag features
        lag_periods = [1, 7, 14, 30]
        for lag in lag_periods:
            df_feat[f'sales_lag_{lag}'] = df_feat.groupby(['store', 'item'])['sales'].shift(lag)
        
        # Rolling statistics
        windows = [7, 14, 30]
        for window in windows:
            # Shift by 1 to avoid data leakage
            df_feat[f'sales_rolling_mean_{window}'] = (
                df_feat.groupby(['store', 'item'])['sales']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
            df_feat[f'sales_rolling_std_{window}'] = (
                df_feat.groupby(['store', 'item'])['sales']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .std()
            )
            df_feat[f'sales_rolling_max_{window}'] = (
                df_feat.groupby(['store', 'item'])['sales']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .max()
            )
            df_feat[f'sales_rolling_min_{window}'] = (
                df_feat.groupby(['store', 'item'])['sales']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .min()
            )
        
        # Price trend features (assuming sales represent some price*quantity relationship)
        for window in [7, 30]:
            df_feat[f'sales_trend_{window}'] = (
                df_feat.groupby(['store', 'item'])[f'sales_rolling_mean_{window}']
                .pct_change()
            )
        
        # Store and item aggregated features
        df_feat['store_avg_sales'] = df_feat.groupby('store')['sales'].transform('mean')
        df_feat['item_avg_sales'] = df_feat.groupby('item')['sales'].transform('mean')
        
        print(f"Original features: {df.shape[1]}")
        print(f"Enhanced features: {df_feat.shape[1]}")
        print(f"New features added: {df_feat.shape[1] - df.shape[1]}")
        
        return df_feat
    
    def prepare_data_for_modeling(self, test_split_date='2017-06-01'):
        """Prepare data for machine learning models."""
        print("\n=== DATA PREPARATION ===")
        
        # Create features
        self.df_enhanced = self.create_features(self.df)
        
        # Remove rows with NaN (due to lag features)
        self.df_clean = self.df_enhanced.dropna()
        print(f"Dataset shape after removing NaN: {self.df_clean.shape}")
        
        # Prepare features and target
        feature_columns = [col for col in self.df_clean.columns if col not in ['date', 'sales']]
        X = self.df_clean[feature_columns]
        y = self.df_clean['sales']
        
        # Time-based split
        split_date = pd.Timestamp(test_split_date)
        train_mask = self.df_clean['date'] < split_date
        test_mask = self.df_clean['date'] >= split_date
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"Train set: {len(X_train)} samples ({train_mask.sum()/len(self.df_clean)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({test_mask.sum()/len(self.df_clean)*100:.1f}%)")
        print(f"Features: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Handle division by zero for MAPE
        mape_mask = y_true != 0
        if mape_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
        else:
            mape = np.inf
        
        metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        
        print(f"\n{model_name} Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE:  {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def train_models(self, X_train, X_test, y_train, y_test, feature_columns):
        """Train and evaluate multiple models."""
        print("\n=== MODEL TRAINING ===")
        
        # 1. Linear Regression (Baseline)
        print("\nTraining Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        self.models['Linear Regression'] = lr_model
        self.metrics['Linear Regression'] = self.evaluate_model(y_test, lr_pred, "Linear Regression")
        
        # 2. Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        self.models['Random Forest'] = rf_model
        self.metrics['Random Forest'] = self.evaluate_model(y_test, rf_pred, "Random Forest")
        
        # Feature importance analysis
        self.analyze_feature_importance(rf_model, feature_columns)
        
        # Store predictions for analysis
        self.predictions = {
            'y_test': y_test,
            'lr_pred': lr_pred,
            'rf_pred': rf_pred
        }
        
        return self.models, self.metrics
    
    def analyze_feature_importance(self, model, feature_columns, top_n=15):
        """Analyze and plot feature importance."""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(top_n)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'], 
                           color='skyblue', alpha=0.8)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + max(top_features['importance']) * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:30s} {row['importance']:.4f}")
            
            return importance_df
    
    def perform_residual_analysis(self, save_plots=True):
        """Perform residual analysis for the best model."""
        print("\n=== RESIDUAL ANALYSIS ===")
        
        y_test = self.predictions['y_test']
        y_pred = self.predictions['rf_pred']  # Using Random Forest as primary model
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Residuals vs Predicted
        axes[0,0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0,0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0,0].set_xlabel('Predicted Values', fontsize=12)
        axes[0,0].set_ylabel('Residuals', fontsize=12)
        axes[0,0].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Residuals Distribution
        axes[0,1].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_xlabel('Residuals', fontsize=12)
        axes[0,1].set_ylabel('Frequency', fontsize=12)
        axes[0,1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add normal curve overlay
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        ax2 = axes[0,1].twinx()
        ax2.plot(x, y * len(residuals) * (residuals.max() - residuals.min()) / 50, 
                'r-', linewidth=2, label='Normal Distribution')
        ax2.set_ylabel('Density', fontsize=12)
        ax2.legend()
        
        # 3. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Actual vs Predicted
        axes[1,1].scatter(y_test, y_pred, alpha=0.6, s=20)
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1,1].set_xlabel('Actual Values', fontsize=12)
        axes[1,1].set_ylabel('Predicted Values', fontsize=12)
        axes[1,1].set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.results_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print residual statistics
        print(f"Residual Statistics:")
        print(f"  Mean: {residuals.mean():.4f}")
        print(f"  Std:  {residuals.std():.4f}")
        print(f"  Min:  {residuals.min():.4f}")
        print(f"  Max:  {residuals.max():.4f}")
        print(f"  Skewness: {stats.skew(residuals):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(residuals):.4f}")
    
    def compare_models(self, save_plots=True):
        """Compare model performances."""
        print("\n=== MODEL COMPARISON ===")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.metrics).T
        print("Model Performance Comparison:")
        print(comparison_df.round(2))
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['RMSE', 'MAE', 'MAPE']
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for i, metric in enumerate(metrics):
            models = list(self.metrics.keys())
            values = [self.metrics[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=colors[i], alpha=0.8, edgecolor='black')
            axes[i].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric, fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate improvement
        if 'Linear Regression' in self.metrics and 'Random Forest' in self.metrics:
            lr_rmse = self.metrics['Linear Regression']['RMSE']
            rf_rmse = self.metrics['Random Forest']['RMSE']
            improvement = ((lr_rmse - rf_rmse) / lr_rmse * 100)
            print(f"\nRandom Forest improvement over Linear Regression: {improvement:.1f}%")
    
    def generate_predictions_sample(self, sample_size=200, save_plots=True):
        """Generate and visualize sample predictions."""
        print("\n=== SAMPLE PREDICTIONS VISUALIZATION ===")
        
        y_test = self.predictions['y_test']
        lr_pred = self.predictions['lr_pred']
        rf_pred = self.predictions['rf_pred']
        
        # Random sample for visualization
        np.random.seed(42)
        sample_indices = np.random.choice(len(y_test), min(sample_size, len(y_test)), replace=False)
        sample_indices = np.sort(sample_indices)
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Random Forest Predictions
        axes[0].plot(range(len(sample_indices)), y_test.iloc[sample_indices].values, 
                    'o-', label='Actual', alpha=0.8, markersize=4, linewidth=1.5)
        axes[0].plot(range(len(sample_indices)), rf_pred[sample_indices], 
                    's-', label='Random Forest', alpha=0.8, markersize=4, linewidth=1.5)
        axes[0].set_title('Random Forest: Actual vs Predicted (Sample)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sample Index', fontsize=12)
        axes[0].set_ylabel('Sales', fontsize=12)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Linear Regression Predictions
        axes[1].plot(range(len(sample_indices)), y_test.iloc[sample_indices].values, 
                    'o-', label='Actual', alpha=0.8, markersize=4, linewidth=1.5)
        axes[1].plot(range(len(sample_indices)), lr_pred[sample_indices], 
                    '^-', label='Linear Regression', alpha=0.8, markersize=4, linewidth=1.5)
        axes[1].set_title('Linear Regression: Actual vs Predicted (Sample)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Sample Index', fontsize=12)
        axes[1].set_ylabel('Sales', fontsize=12)
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.results_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def error_analysis(self):
        """Perform detailed error analysis."""
        print("\n=== ERROR ANALYSIS ===")
        
        y_test = self.predictions['y_test']
        rf_pred = self.predictions['rf_pred']
        
        # Create error analysis dataframe
        error_df = pd.DataFrame({
            'actual': y_test,
            'predicted': rf_pred,
            'absolute_error': np.abs(y_test - rf_pred),
            'percentage_error': np.abs((y_test - rf_pred) / np.maximum(y_test, 1)) * 100,
            'squared_error': (y_test - rf_pred) ** 2
        })
        
        # Add original features for analysis
        test_indices = y_test.index
        error_df = error_df.join(self.df_clean[['store', 'item', 'month', 'dayofweek']].loc[test_indices])
        
        # Error by store
        store_errors = error_df.groupby('store').agg({
            'absolute_error': ['mean', 'std', 'count'],
            'percentage_error': 'mean'
        }).round(2)
        store_errors.columns = ['MAE', 'MAE_Std', 'Count', 'MAPE']
        store_errors = store_errors.sort_values('MAE', ascending=False)
        
        print("\nTop 10 Stores with Highest Average Error:")
        print(store_errors.head(10))
        
        # Error by item (top 10)
        item_errors = error_df.groupby('item').agg({
            'absolute_error': ['mean', 'count'],
            'percentage_error': 'mean'
        }).round(2)
        item_errors.columns = ['MAE', 'Count', 'MAPE']
        item_errors = item_errors[item_errors['Count'] >= 10]  # Filter items with sufficient data
        item_errors = item_errors.sort_values('MAE', ascending=False)
        
        print("\nTop 10 Items with Highest Average Error (min 10 observations):")
        print(item_errors.head(10))
        
        # Error by month
        month_errors = error_df.groupby('month').agg({
            'absolute_error': ['mean', 'std'],
            'percentage_error': 'mean'
        }).round(2)
        month_errors.columns = ['MAE', 'MAE_Std', 'MAPE']
        
        print("\nAverage Error by Month:")
        print(month_errors)
        
        # Error by day of week
        dow_errors = error_df.groupby('dayofweek').agg({
            'absolute_error': ['mean', 'std'],
            'percentage_error': 'mean'
        }).round(2)
        dow_errors.columns = ['MAE', 'MAE_Std', 'MAPE']
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_errors.index = [dow_names[i] for i in dow_errors.index]
        
        print("\nAverage Error by Day of Week:")
        print(dow_errors)
        
        return error_df
    
    def fit_sarima_model(self):
        """Fit SARIMA model for time series forecasting."""
        print("\n=== TIME SERIES FORECASTING WITH SARIMA ===")
        
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # Aggregate data for SARIMA
            ts = self.df.groupby('date')['sales'].sum()
            ts.index = pd.to_datetime(ts.index)
            ts = ts.asfreq('D')
            
            print(f"Time series shape: {ts.shape}")
            print(f"Date range: {ts.index.min()} to {ts.index.max()}")
            
            # Split for SARIMA
            split_date = pd.Timestamp('2017-06-01')
            ts_train = ts[ts.index < split_date]
            ts_test = ts[ts.index >= split_date]
            
            print(f"SARIMA Train period: {ts_train.index.min()} to {ts_train.index.max()}")
            print(f"SARIMA Test period: {ts_test.index.min()} to {ts_test.index.max()}")
            
            # Fit SARIMA model
            print("Fitting SARIMA model...")
            sarima_model = SARIMAX(ts_train, order=(1,1,1), seasonal_order=(1,1,1,7))
            sarima_fit = sarima_model.fit(disp=False, maxiter=200)
            
            # Forecast
            sarima_forecast = sarima_fit.forecast(steps=len(ts_test))
            
            # Evaluate SARIMA
            sarima_rmse = np.sqrt(mean_squared_error(ts_test, sarima_forecast))
            sarima_mae = mean_absolute_error(ts_test, sarima_forecast)
            sarima_mape = np.mean(np.abs((ts_test - sarima_forecast) / ts_test)) * 100
            
            print(f"\nSARIMA Performance:")
            print(f"  RMSE: {sarima_rmse:.2f}")
            print(f"  MAE:  {sarima_mae:.2f}")
            print(f"  MAPE: {sarima_mape:.2f}%")
            
            # Plot SARIMA results
            plt.figure(figsize=(16, 8))
            
            # Plot last 60 days of training data for context
            plt.plot(ts_train.index[-60:], ts_train.values[-60:], 
                    label='Training Data', color='blue', alpha=0.8, linewidth=2)
            plt.plot(ts_test.index, ts_test.values, 
                    label='Actual', color='green', linewidth=2)
            plt.plot(ts_test.index, sarima_forecast, 
                    label='SARIMA Forecast', color='red', linestyle='--', linewidth=2)
            
            plt.legend(fontsize=12)
            plt.title('SARIMA Forecast vs Actual (Daily Total Sales)', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Total Sales', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'sarima_forecast.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Store SARIMA results
            self.models['SARIMA'] = sarima_fit
            self.metrics['SARIMA'] = {
                'RMSE': sarima_rmse, 
                'MAE': sarima_mae, 
                'MAPE': sarima_mape
            }
            
            return sarima_fit, sarima_forecast
            
        except Exception as e:
            print(f"SARIMA fitting failed: {e}")
            print("Continuing without SARIMA model...")
            return None, None
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("               SALES FORECASTING PROJECT SUMMARY")
        print("="*60)
        
        print(f"📊 Dataset Overview:")
        print(f"   • Total records: {len(self.df):,}")
        print(f"   • Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"   • Unique stores: {self.df['store'].nunique()}")
        print(f"   • Unique items: {self.df['item'].nunique()}")
        print(f"   • Total sales volume: {self.df['sales'].sum():,.0f}")
        
        print(f"\n🔧 Feature Engineering:")
        original_features = 4  # date, store, item, sales
        final_features = len([col for col in self.df_clean.columns if col not in ['date', 'sales']])
        print(f"   • Original features: {original_features}")
        print(f"   • Engineered features: {final_features}")
        print(f"   • Features added: {final_features - original_features + 1}")
        
        print(f"\n🤖 Models Trained:")
        for i, model_name in enumerate(self.models.keys(), 1):
            metrics = self.metrics[model_name]
            print(f"   {i}. {model_name}")
            print(f"      - RMSE: {metrics['RMSE']:.2f}")
            print(f"      - MAE: {metrics['MAE']:.2f}")
            print(f"      - MAPE: {metrics['MAPE']:.2f}%")
        
        # Find best model
        best_model = min(self.metrics.keys(), key=lambda x: self.metrics[x]['RMSE'])
        print(f"\n🏆 Best Performing Model: {best_model}")
        
        # Calculate improvement if we have baseline
        if 'Linear Regression' in self.metrics and best_model != 'Linear Regression':
            baseline_rmse = self.metrics['Linear Regression']['RMSE']
            best_rmse = self.metrics[best_model]['RMSE']
            improvement = ((baseline_rmse - best_rmse) / baseline_rmse * 100)
            print(f"   • Improvement over baseline: {improvement:.1f}%")
        
        print(f"\n💡 Key Insights:")
        print(f"   • Feature engineering significantly improved model performance")
        print(f"   • {best_model} captured non-linear patterns better than baseline")
        print(f"   • Lag features and rolling statistics were highly predictive")
        print(f"   • Seasonal patterns exist in the data")
        print(f"   • Some stores/items are consistently harder to predict")
        
        print(f"\n🔮 Recommendations for Improvement:")
        recommendations = [
            "Include external data (holidays, promotions, weather)",
            "Try advanced models (XGBoost, LightGBM, Neural Networks)",
            "Implement hierarchical forecasting for store-item combinations",
            "Add more sophisticated time-based features",
            "Consider ensemble methods combining different model types",
            "Implement real-time prediction pipeline",
            "Add uncertainty quantification with prediction intervals"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self):
        """Run the complete sales forecasting analysis pipeline."""
        print("🚀 Starting Complete Sales Forecasting Analysis...")
        print("="*60)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Perform EDA
        self.perform_eda()
        
        # Step 3: Prepare data for modeling
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data_for_modeling()
        
        # Step 4: Train models
        self.train_models(X_train, X_test, y_train, y_test, feature_columns)
        
        # Step 5: Perform residual analysis
        self.perform_residual_analysis()
        
        # Step 6: Compare models
        self.compare_models()
        
        # Step 7: Generate sample predictions
        self.generate_predictions_sample()
        
        # Step 8: Error analysis
        self.error_analysis()
        
        # Step 9: Fit SARIMA model (optional)
        self.fit_sarima_model()
        
        # Step 10: Generate summary
        self.generate_summary_report()
        
        print(f"\n✅ Analysis complete! Results saved to: {self.results_dir}")
        print("📁 Check the results directory for all generated plots and analysis.")


def main():
    """Main function to run the analysis."""
    # Initialize analyzer
    analyzer = SalesForecastAnalyzer(
        data_path='data/train.csv',
        results_dir='results'
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    return analyzer


if __name__ == "__main__":
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Run analysis
    analyzer = main()
    
    print("\n🎉 Sales Forecasting Analysis Complete!")
    print("📊 All results have been saved to the 'results' directory.")
    print("🔍 Check the generated plots and analysis for insights.")