#!/usr/bin/env python3
"""
COMPREHENSIVE MAMMOGRAPHY HISTORY STUDY
=======================================
This script runs multiple iterations with different history lengths,
saves all results, and provides comprehensive comparison.

Research Question: Does more mammography history improve breast cancer prediction?

Approach:
1. Train models with HISTORY = 0, 1, 2, 3, 4
2. Save predictions and metrics for each
3. Provide comprehensive comparison analysis
4. Generate publication-ready results tables

Author: Research Team
Date: September 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import time
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("="*80)
print("COMPREHENSIVE MAMMOGRAPHY HISTORY STUDY")
print("="*80)
print("This will run all history lengths (0-4 years) and compare results")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ===========================================================================================
# CONFIGURATION
# ===========================================================================================

class Config:
    """Configuration parameters"""
    # Dataset paths
    METADATA_PATH = "/kaggle/input/breast-cancer-research-metadata/CSAW-CC_breast_cancer_screening_data.csv"
    LOCAL_METADATA_PATH = "CSAW-CC_breast_cancer_screening_data.csv"
    
    # History configurations to test
    HISTORY_CONFIGS = [0, 1, 2, 3, 4]
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Model parameters
    DROPOUT_RATE = 0.3
    
    # Data split parameters (keep same splits across all history lengths)
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Evaluation parameters
    N_BOOTSTRAP = 1000

config = Config()

# ===========================================================================================
# HELPER FUNCTIONS
# ===========================================================================================

def concordance_index(y_true, y_pred):
    """Calculate C-index (concordance index)"""
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def bootstrap_ci(y_true, y_pred, metric_func=roc_auc_score, n_bootstrap=1000, alpha=0.05):
    """Calculate bootstrap confidence intervals"""
    if len(np.unique(y_true)) <= 1:
        return 0.5, [0.5, 0.5]
    
    try:
        original_metric = metric_func(y_true, y_pred)
        
        bootstrap_metrics = []
        n_samples = len(y_true)
        
        for _ in range(min(n_bootstrap, 100)):  # Reduced for speed
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            if len(np.unique(y_true_boot)) > 1:
                boot_metric = metric_func(y_true_boot, y_pred_boot)
                bootstrap_metrics.append(boot_metric)
        
        if bootstrap_metrics:
            ci_lower = np.percentile(bootstrap_metrics, 100 * alpha/2)
            ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha/2))
            return original_metric, [ci_lower, ci_upper]
        else:
            return original_metric, [original_metric, original_metric]
            
    except Exception as e:
        print(f"Error in bootstrap CI: {e}")
        return 0.5, [0.5, 0.5]

# ===========================================================================================
# DATA LOADING AND PREPROCESSING
# ===========================================================================================

def load_real_data():
    """Load the real mammography dataset"""
    print("\nLoading real mammography dataset...")
    
    # Try to load from different possible paths
    for path in [config.METADATA_PATH, config.LOCAL_METADATA_PATH]:
        try:
            df = pd.read_csv(path, low_memory=False)
            print(f"Loaded dataset from {path}")
            print(f"Dataset shape: {df.shape}")
            return df
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError("Could not find the dataset file")

def preprocess_patient_history_real(df, history_years=0):
    """
    Preprocess real patient data for history analysis
    """
    print(f"\nProcessing real patient data for {history_years} years of history...")
    
    # Check available columns
    print(f"Available columns: {df.columns.tolist()}")
    
    # Clean data
    df_clean = df.copy()
    
    # Ensure we have required columns
    required_cols = ['anon_patientid', 'exam_year', 'x_case']
    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()  # Return empty if missing critical columns
    
    # Remove rows with missing critical data
    df_clean = df_clean.dropna(subset=['anon_patientid', 'exam_year', 'x_case'])
    print(f"After removing missing critical data: {len(df_clean)} records")
    
    # Sort by patient and exam year
    df_clean = df_clean.sort_values(['anon_patientid', 'exam_year'])
    
    # Group by patient
    patient_groups = df_clean.groupby('anon_patientid')
    processed_patients = []
    
    print(f"Processing {len(patient_groups)} patients...")
    
    for patient_id, patient_data in patient_groups:
        # Sort by exam year (most recent first)
        patient_data = patient_data.sort_values('exam_year', ascending=False)
        
        # Need at least 1 exam
        if len(patient_data) == 0:
            continue
        
        # For the current approach, use the most recent exam as "current"
        # and take up to history_years previous exams
        required_exams = history_years + 1
        available_exams = len(patient_data)
        
        if available_exams < 1:
            continue
        
        # Take available exams up to required number
        selected_exams = patient_data.head(min(required_exams, available_exams))
        
        # Current exam (most recent)
        current_exam = selected_exams.iloc[0]
        
        # Create features for this patient
        patient_features = {
            'patient_id': patient_id,
            'current_exam_year': current_exam['exam_year'],
            'cancer_outcome': current_exam['x_case'],
            'num_historical_exams': len(selected_exams) - 1,
            'years_of_data': current_exam['exam_year'] - selected_exams.iloc[-1]['exam_year'] if len(selected_exams) > 1 else 0
        }
        
        # Add mammographic features if available
        mammo_features = ['libra_percentdensity', 'libra_breastarea', 'libra_densearea', 
                         'libra_nondensearea', 'age_at_study']
        
        for feature in mammo_features:
            if feature in selected_exams.columns:
                values = selected_exams[feature].dropna()
                if len(values) > 0:
                    patient_features[f'{feature}_current'] = values.iloc[0]
                    
                    if len(values) > 1:
                        patient_features[f'{feature}_mean'] = values.mean()
                        patient_features[f'{feature}_std'] = values.std()
                        patient_features[f'{feature}_trend'] = values.iloc[0] - values.iloc[-1]
                        patient_features[f'{feature}_change_per_year'] = patient_features[f'{feature}_trend'] / max(1, patient_features['years_of_data'])
                    else:
                        patient_features[f'{feature}_mean'] = values.iloc[0]
                        patient_features[f'{feature}_std'] = 0.0
                        patient_features[f'{feature}_trend'] = 0.0
                        patient_features[f'{feature}_change_per_year'] = 0.0
        
        # Add derived features based on history length
        if history_years > 0 and len(selected_exams) > 1:
            # Temporal stability features
            if 'libra_percentdensity' in selected_exams.columns:
                density_values = selected_exams['libra_percentdensity'].dropna()
                if len(density_values) > 1:
                    patient_features['density_volatility'] = density_values.std()
                    patient_features['density_trend_direction'] = 1 if density_values.iloc[0] > density_values.iloc[-1] else 0
        
        processed_patients.append(patient_features)
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_patients)
    
    if len(processed_df) > 0:
        print(f"Processed {len(processed_df)} patients with {history_years} years of history")
        print(f"Cancer cases: {processed_df['cancer_outcome'].sum()}/{len(processed_df)} ({100*processed_df['cancer_outcome'].mean():.1f}%)")
        
        # Print feature summary
        feature_cols = [col for col in processed_df.columns if col not in ['patient_id', 'cancer_outcome', 'current_exam_year']]
        print(f"Generated {len(feature_cols)} features")
        
    return processed_df

def prepare_features_real(df):
    """Prepare features from real processed data"""
    print("\nPreparing features for model training...")
    
    if len(df) == 0:
        return df, [], {}
    
    # Identify feature columns
    exclude_cols = ['patient_id', 'cancer_outcome', 'current_exam_year']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate numerical and categorical features
    numerical_features = []
    categorical_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Prepare final DataFrame
    feature_df = df.copy()
    
    # Handle missing values in numerical features
    for col in numerical_features:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    
    # Encode categorical features
    le_dict = {}
    encoded_features = []
    for col in categorical_features:
        le = LabelEncoder()
        feature_df[col] = feature_df[col].fillna('Unknown').astype(str)
        encoded_col = f'{col}_encoded'
        feature_df[encoded_col] = le.fit_transform(feature_df[col])
        le_dict[col] = le
        encoded_features.append(encoded_col)
    
    # Final feature list
    final_features = numerical_features + encoded_features
    
    print(f"Final features ({len(final_features)}): {final_features}")
    
    return feature_df, final_features, le_dict

# ===========================================================================================
# MODEL ARCHITECTURE
# ===========================================================================================

class BreastCancerPredictor(nn.Module):
    """Neural network for breast cancer prediction"""
    
    def __init__(self, num_features, dropout_rate=0.3):
        super(BreastCancerPredictor, self).__init__()
        
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.feature_processor(x)

class BreastCancerDataset(Dataset):
    """Dataset class for breast cancer prediction"""
    
    def __init__(self, df, feature_columns, target_column='cancer_outcome'):
        self.features = df[feature_columns].values.astype(np.float32)
        self.targets = df[target_column].values.astype(np.float32)
        
        # Handle NaN values
        self.features = np.nan_to_num(self.features, nan=0.0)
        
        print(f"Dataset created: {len(self.features)} samples, {len(feature_columns)} features")
        print(f"Target distribution: {np.bincount(self.targets.astype(int))}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])

# ===========================================================================================
# TRAINING AND EVALUATION
# ===========================================================================================

def train_model_iteration(model, train_loader, val_loader, device, num_epochs=20):
    """Train model for one iteration"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if epoch % 5 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model_iteration(model, test_loader, device):
    """Evaluate model and return predictions"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features).squeeze()
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)

# ===========================================================================================
# MAIN STUDY EXECUTION
# ===========================================================================================

def run_comprehensive_study():
    """Run comprehensive study across all history lengths"""
    
    print(f"\n{'='*80}")
    print("STARTING COMPREHENSIVE MAMMOGRAPHY HISTORY STUDY")
    print(f"{'='*80}")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load real data
    try:
        df = load_real_data()
    except FileNotFoundError:
        print("Real dataset not found. Cannot proceed with real data analysis.")
        return
    
    # Store all results
    all_results = {}
    all_predictions = {}
    
    # Fixed data splits (same patients in train/val/test across all history lengths)
    print("\nCreating consistent data splits across all history lengths...")
    
    # Get unique patient IDs for consistent splitting
    unique_patients = df['anon_patientid'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train_patients, val_patients = train_test_split(
        train_patients, test_size=config.VAL_SIZE/(1-config.TEST_SIZE), random_state=config.RANDOM_STATE
    )
    
    print(f"Patient splits - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    
    # Run analysis for each history configuration
    for history_years in config.HISTORY_CONFIGS:
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL WITH {history_years} YEARS OF HISTORY")
        print(f"{'='*80}")
        
        # Process data for this history length
        processed_df = preprocess_patient_history_real(df, history_years)
        
        if len(processed_df) == 0:
            print(f"No data available for {history_years} years of history. Skipping...")
            continue
        
        # Prepare features
        feature_df, feature_columns, le_dict = prepare_features_real(processed_df)
        
        if len(feature_columns) == 0:
            print(f"No features available for {history_years} years of history. Skipping...")
            continue
        
        # Split data using consistent patient splits
        train_df = feature_df[feature_df['patient_id'].isin(train_patients)]
        val_df = feature_df[feature_df['patient_id'].isin(val_patients)]
        test_df = feature_df[feature_df['patient_id'].isin(test_patients)]
        
        print(f"Data splits for {history_years} years:")
        print(f"  Train: {len(train_df)} samples ({train_df['cancer_outcome'].sum()} positive)")
        print(f"  Val:   {len(val_df)} samples ({val_df['cancer_outcome'].sum()} positive)")
        print(f"  Test:  {len(test_df)} samples ({test_df['cancer_outcome'].sum()} positive)")
        
        # Skip if insufficient data
        if len(train_df) < 10 or len(test_df) < 10:
            print(f"Insufficient data for {history_years} years of history. Skipping...")
            continue
        
        # Create datasets and loaders
        train_dataset = BreastCancerDataset(train_df, feature_columns)
        val_dataset = BreastCancerDataset(val_df, feature_columns)
        test_dataset = BreastCancerDataset(test_df, feature_columns)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Train model
        model = BreastCancerPredictor(len(feature_columns), config.DROPOUT_RATE)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        trained_model = train_model_iteration(model, train_loader, val_loader, device, config.NUM_EPOCHS)
        
        # Evaluate model
        predictions, targets = evaluate_model_iteration(trained_model, test_loader, device)
        
        # Calculate metrics
        if len(np.unique(targets)) > 1:
            auc, auc_ci = bootstrap_ci(targets, predictions, roc_auc_score)
            c_index, c_index_ci = bootstrap_ci(targets, predictions, concordance_index)
        else:
            auc, auc_ci = 0.5, [0.5, 0.5]
            c_index, c_index_ci = 0.5, [0.5, 0.5]
        
        # Store results
        results = {
            'history_years': history_years,
            'n_features': len(feature_columns),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'auc': auc,
            'auc_ci_lower': auc_ci[0],
            'auc_ci_upper': auc_ci[1],
            'c_index': c_index,
            'c_index_ci_lower': c_index_ci[0],
            'c_index_ci_upper': c_index_ci[1],
            'feature_columns': feature_columns,
            'timestamp': datetime.now().isoformat()
        }
        
        all_results[f'history_{history_years}'] = results
        all_predictions[f'history_{history_years}'] = {
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'patient_ids': test_df['patient_id'].tolist()
        }
        
        # Print results for this iteration
        print(f"\nResults for {history_years} years of history:")
        print(f"  Features: {len(feature_columns)}")
        print(f"  AUC: {auc:.4f} (95% CI: {auc_ci[0]:.4f}-{auc_ci[1]:.4f})")
        print(f"  C-Index: {c_index:.4f} (95% CI: {c_index_ci[0]:.4f}-{c_index_ci[1]:.4f})")
    
    # Save all results
    save_comprehensive_results(all_results, all_predictions)
    
    # Generate comparison analysis
    generate_comparison_analysis(all_results)
    
    return all_results, all_predictions

def save_comprehensive_results(all_results, all_predictions):
    """Save all results to files"""
    
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save summary results
    summary_data = []
    for config_name, results in all_results.items():
        summary_data.append({
            'History_Years': results['history_years'],
            'Features': results['n_features'],
            'Train_Samples': results['n_train'],
            'Test_Samples': results['n_test'],
            'AUC': results['auc'],
            'AUC_CI_Lower': results['auc_ci_lower'],
            'AUC_CI_Upper': results['auc_ci_upper'],
            'C_Index': results['c_index'],
            'C_Index_CI_Lower': results['c_index_ci_lower'],
            'C_Index_CI_Upper': results['c_index_ci_upper']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('mammography_history_study_summary.csv', index=False)
    print("Summary results saved to: mammography_history_study_summary.csv")
    
    # Save detailed results
    with open('mammography_history_study_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Detailed results saved to: mammography_history_study_detailed.json")
    
    # Save predictions
    with open('mammography_history_study_predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print("Predictions saved to: mammography_history_study_predictions.json")

def generate_comparison_analysis(all_results):
    """Generate comprehensive comparison analysis"""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    if len(all_results) < 2:
        print("Need at least 2 history configurations for comparison")
        return
    
    # Extract data for analysis
    history_years = []
    aucs = []
    c_indices = []
    n_features = []
    
    for config_name in sorted(all_results.keys()):
        results = all_results[config_name]
        history_years.append(results['history_years'])
        aucs.append(results['auc'])
        c_indices.append(results['c_index'])
        n_features.append(results['n_features'])
    
    # Results table
    print("\nRESULTS SUMMARY TABLE")
    print("-" * 80)
    print(f"{'History':<8} {'Features':<10} {'AUC':<8} {'95% CI':<20} {'C-Index':<8} {'95% CI':<20}")
    print("-" * 80)
    
    for config_name in sorted(all_results.keys()):
        results = all_results[config_name]
        h_years = results['history_years']
        n_feat = results['n_features']
        auc = results['auc']
        auc_ci = f"({results['auc_ci_lower']:.3f}-{results['auc_ci_upper']:.3f})"
        c_idx = results['c_index']
        c_ci = f"({results['c_index_ci_lower']:.3f}-{results['c_index_ci_upper']:.3f})"
        
        print(f"{h_years:<8} {n_feat:<10} {auc:<8.4f} {auc_ci:<20} {c_idx:<8.4f} {c_ci:<20}")
    
    # Performance trend analysis
    print("\nPERFORMANCE TREND ANALYSIS")
    print("-" * 60)
    
    if len(history_years) > 1:
        auc_trend = aucs[-1] - aucs[0]
        feature_trend = n_features[-1] - n_features[0]
        
        print(f"AUC change from {history_years[0]} to {history_years[-1]} years: {auc_trend:+.4f}")
        print(f"Feature count change: {feature_trend:+d}")
        
        # Find best performing configuration
        best_idx = np.argmax(aucs)
        best_config = history_years[best_idx]
        best_auc = aucs[best_idx]
        
        print(f"Best performing configuration: {best_config} years (AUC = {best_auc:.4f})")
        
        # Statistical significance (simple check)
        if len(aucs) > 2:
            improvement_threshold = 0.01  # 1% improvement
            significant_improvements = []
            
            for i in range(1, len(aucs)):
                improvement = aucs[i] - aucs[0]  # vs baseline (0 years)
                if improvement > improvement_threshold:
                    significant_improvements.append((history_years[i], improvement))
            
            if significant_improvements:
                print(f"Configurations with >1% improvement over baseline:")
                for years, improvement in significant_improvements:
                    print(f"  {years} years: +{improvement:.4f} AUC")
            else:
                print("No configurations showed >1% improvement over baseline")
    
    # Create visualization
    create_comparison_plots(history_years, aucs, c_indices, n_features, all_results)

def create_comparison_plots(history_years, aucs, c_indices, n_features, all_results):
    """Create comprehensive comparison plots"""
    
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: AUC vs History Length
        axes[0, 0].plot(history_years, aucs, 'o-', linewidth=2, markersize=8, color='blue')
        axes[0, 0].set_xlabel('Years of History')
        axes[0, 0].set_ylabel('AUC (Area Under Curve)')
        axes[0, 0].set_title('AUC vs Mammography History Length')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0.4, 1.0])
        
        # Add confidence intervals
        for i, config_name in enumerate(sorted(all_results.keys())):
            results = all_results[config_name]
            ci_lower = results['auc_ci_lower']
            ci_upper = results['auc_ci_upper']
            axes[0, 0].errorbar(results['history_years'], results['auc'], 
                              yerr=[[results['auc'] - ci_lower], [ci_upper - results['auc']]], 
                              fmt='o', capsize=5, alpha=0.7)
        
        # Plot 2: C-Index vs History Length
        axes[0, 1].plot(history_years, c_indices, 's-', linewidth=2, markersize=8, color='red')
        axes[0, 1].set_xlabel('Years of History')
        axes[0, 1].set_ylabel('C-Index (Concordance Index)')
        axes[0, 1].set_title('C-Index vs Mammography History Length')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.4, 1.0])
        
        # Plot 3: Number of Features vs History Length
        axes[1, 0].bar(history_years, n_features, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Years of History')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].set_title('Feature Count vs History Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance vs Features
        axes[1, 1].scatter(n_features, aucs, s=100, alpha=0.7, color='purple')
        for i, years in enumerate(history_years):
            axes[1, 1].annotate(f'{years}y', (n_features[i], aucs[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('AUC vs Number of Features')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mammography_history_study_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nComprehensive plots saved as: mammography_history_study_comprehensive.png")
        
    except Exception as e:
        print(f"Error creating plots: {e}")

# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================

if __name__ == "__main__":
    print("Starting comprehensive mammography history study...")
    
    start_time = time.time()
    results, predictions = run_comprehensive_study()
    end_time = time.time()
    
    print(f"\n{'='*80}")
    print("STUDY COMPLETE!")
    print(f"{'='*80}")
    print(f"Total execution time: {(end_time - start_time)/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results:
        print(f"\nProcessed {len(results)} history configurations")
        print("Results saved to CSV and JSON files")
        print("Comprehensive analysis and plots generated")
        
        print(f"\nFINAL ANSWER TO RESEARCH QUESTION:")
        print(f"{'='*50}")
        
        history_years = [results[k]['history_years'] for k in sorted(results.keys())]
        aucs = [results[k]['auc'] for k in sorted(results.keys())]
        
        if len(aucs) > 1:
            if aucs[-1] > aucs[0]:
                print(f"✅ YES - More history IS better!")
                print(f"   AUC improved from {aucs[0]:.4f} to {aucs[-1]:.4f}")
                print(f"   Improvement: {aucs[-1] - aucs[0]:+.4f}")
            else:
                print(f"❌ NO - More history does NOT improve performance")
                print(f"   AUC decreased from {aucs[0]:.4f} to {aucs[-1]:.4f}")
                print(f"   Change: {aucs[-1] - aucs[0]:+.4f}")
        
        best_config = max(results.keys(), key=lambda k: results[k]['auc'])
        best_years = results[best_config]['history_years']
        best_auc = results[best_config]['auc']
        print(f"🏆 BEST CONFIGURATION: {best_years} years (AUC = {best_auc:.4f})")
    else:
        print("\nNo results generated. Check data availability and processing.")
