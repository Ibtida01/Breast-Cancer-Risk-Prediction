#!/usr/bin/env python3
"""
REAL MAMMOGRAPHY HISTORY ANALYSIS: USING ACTUAL PATIENT DATA
===========================================================
Research Question: Does more mammography history improve breast cancer prediction?

This script analyzes REAL patient history from the CSAW-CC dataset:
- Uses actual patient IDs (anon_patientid)  
- Uses actual exam years (exam_year)
- Uses actual mammography features (breast density, area measurements)
- Uses actual cancer outcomes (x_case)

History configurations:
- HISTORY = 0: Current exam only
- HISTORY = 1: Current + 1 previous exam (up to 2 exams)
- HISTORY = 2: Current + 2 previous exams (up to 3 exams)
- HISTORY = 3: Current + 3 previous exams (up to 4 exams)
- HISTORY = 4: Current + 4 previous exams (up to 5 exams)

Author: Research Team
Date: September 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ===========================================================================================
# CONFIGURATION - SET HISTORY HERE
# ===========================================================================================

# **MAIN PARAMETER TO ADJUST**
HISTORY = 0  # Change this value from 0 to 4 for different experiments

class Config:
    """Configuration parameters"""
    # Dataset path
    METADATA_PATH = "/home/schatten/Documents/B-C-Research/Transformer Model/CSAW-CC_breast_cancer_screening_data.csv"
    
    # History configuration
    HISTORY_YEARS = HISTORY  # Number of historical exams to include (0-4)
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    
    # Model parameters  
    DROPOUT_RATE = 0.3
    
    # Data split parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    # Evaluation parameters
    N_BOOTSTRAP = 1000

config = Config()

print("="*80)
print("REAL MAMMOGRAPHY HISTORY ANALYSIS")
print("="*80)
print(f"HISTORY SETTING: {HISTORY} previous exams")
print(f"- Training data: Current + {HISTORY} previous exams = {HISTORY + 1} total exams per patient")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ===========================================================================================
# HELPER FUNCTIONS
# ===========================================================================================

def concordance_index(y_true, y_pred):
    """Calculate C-index (concordance index) - equivalent to AUC for binary classification"""
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def bootstrap_ci(y_true, y_pred, metric_func=roc_auc_score, n_bootstrap=1000, alpha=0.05):
    """Calculate bootstrap confidence intervals for a metric"""
    if len(np.unique(y_true)) <= 1:
        return 0.5, [0.5, 0.5]
    
    try:
        # Original metric
        original_metric = metric_func(y_true, y_pred)
        
        # Bootstrap
        bootstrap_metrics = []
        n_samples = len(y_true)
        
        for _ in range(min(100, n_bootstrap)):  # Reduced for speed
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Skip if no positive cases
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
# REAL DATA PREPROCESSING
# ===========================================================================================

def load_and_analyze_real_data(metadata_path, history_years=0):
    """
    Load and analyze REAL patient data with actual mammography history
    """
    print(f"\nLoading real patient data for {history_years} previous exams...")
    
    # Load the real dataset
    df = pd.read_csv(metadata_path, low_memory=False)
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic data info
    print(f"\nDataset overview:")
    print(f"  Unique patients: {df['anon_patientid'].nunique()}")
    print(f"  Exam years range: {df['exam_year'].min()} - {df['exam_year'].max()}")
    print(f"  Cancer cases: {df['x_case'].sum()} / {len(df)} ({100*df['x_case'].mean():.1f}%)")
    print(f"  Image views: {df['viewposition'].value_counts().to_dict()}")
    print(f"  Image laterality: {df['imagelaterality'].value_counts().to_dict()}")
    
    return df

def create_patient_history_features(df, history_years=0):
    """
    Create features using real patient history data
    """
    print(f"\nCreating patient history features with {history_years} previous exams...")
    
    # Sort by patient and exam year
    df_sorted = df.sort_values(['anon_patientid', 'exam_year']).reset_index(drop=True)
    
    # Group by patient to get their exam history
    patient_features = []
    patients_processed = 0
    patients_skipped = 0
    
    for patient_id, patient_data in df_sorted.groupby('anon_patientid'):
        # Sort by exam year (most recent first for consistent ordering)
        patient_exams = patient_data.sort_values('exam_year', ascending=False)
        
        # We need at least 1 exam
        if len(patient_exams) == 0:
            patients_skipped += 1
            continue
            
        # For each exam, create features using that exam as "current" and previous as history
        for exam_idx in range(len(patient_exams)):
            current_exam = patient_exams.iloc[exam_idx]
            
            # Get historical exams (those BEFORE the current exam year)
            historical_exams = patient_exams[patient_exams['exam_year'] < current_exam['exam_year']]
            
            # Check if we have enough history
            available_history = len(historical_exams)
            if available_history < history_years:
                # Skip if we don't have enough history for this configuration
                continue
            
            # Take only the required number of historical exams (most recent ones)
            if history_years > 0:
                selected_history = historical_exams.head(history_years)
                all_exams = pd.concat([current_exam.to_frame().T, selected_history], ignore_index=True)
            else:
                all_exams = current_exam.to_frame().T
            
            # Extract features from current and historical exams
            features = extract_mammography_features(all_exams, current_exam, history_years)
            patient_features.append(features)
        
        patients_processed += 1
        if patients_processed % 1000 == 0:
            print(f"  Processed {patients_processed} patients...")
    
    print(f"Processed {patients_processed} patients, skipped {patients_skipped}")
    print(f"Created {len(patient_features)} patient-exam combinations")
    
    if len(patient_features) == 0:
        raise ValueError(f"No patient data available for history_years={history_years}")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(patient_features)
    
    print(f"Features created: {len(features_df)} samples")
    print(f"Cancer cases: {features_df['cancer_outcome'].sum()} / {len(features_df)} ({100*features_df['cancer_outcome'].mean():.1f}%)")
    
    return features_df

def extract_mammography_features(all_exams, current_exam, history_years):
    """
    Extract meaningful features from current and historical mammography exams
    """
    features = {
        'patient_id': current_exam['anon_patientid'],
        'current_exam_year': current_exam['exam_year'],
        'cancer_outcome': int(current_exam['x_case']),
        'history_length': len(all_exams),
    }
    
    # Current exam features
    current_features = extract_single_exam_features(current_exam, 'current')
    features.update(current_features)
    
    if history_years > 0 and len(all_exams) > 1:
        # Historical exam features (aggregate)
        historical_exams = all_exams.iloc[1:]  # Skip current exam
        
        # Aggregate historical features
        density_values = []
        area_values = []
        
        for _, exam in historical_exams.iterrows():
            if pd.notna(exam['libra_percentdensity']):
                density_values.append(exam['libra_percentdensity'])
            if pd.notna(exam['libra_breastarea']):
                area_values.append(exam['libra_breastarea'])
        
        if density_values:
            features.update({
                'historical_density_mean': np.mean(density_values),
                'historical_density_std': np.std(density_values) if len(density_values) > 1 else 0,
                'historical_density_min': np.min(density_values),
                'historical_density_max': np.max(density_values),
                'historical_density_trend': density_values[0] - density_values[-1] if len(density_values) > 1 else 0,
            })
        else:
            features.update({
                'historical_density_mean': 0,
                'historical_density_std': 0,
                'historical_density_min': 0,
                'historical_density_max': 0,
                'historical_density_trend': 0,
            })
        
        if area_values:
            features.update({
                'historical_area_mean': np.mean(area_values),
                'historical_area_std': np.std(area_values) if len(area_values) > 1 else 0,
                'historical_area_trend': area_values[0] - area_values[-1] if len(area_values) > 1 else 0,
            })
        else:
            features.update({
                'historical_area_mean': 0,
                'historical_area_std': 0,
                'historical_area_trend': 0,
            })
        
        # Time span features
        exam_years = all_exams['exam_year'].tolist()
        features.update({
            'time_span_years': max(exam_years) - min(exam_years),
            'years_since_last_exam': 0,  # Current is always 0
            'exam_frequency': len(exam_years) / max(1, max(exam_years) - min(exam_years)) if len(set(exam_years)) > 1 else 1,
        })
        
        # Density change features (current vs historical)
        if density_values and pd.notna(current_exam['libra_percentdensity']):
            current_density = current_exam['libra_percentdensity']
            features.update({
                'density_change_from_last': current_density - density_values[0] if len(density_values) > 0 else 0,
                'density_change_from_mean': current_density - np.mean(density_values),
                'density_volatility': np.std(density_values + [current_density]) if len(density_values) > 0 else 0,
            })
        else:
            features.update({
                'density_change_from_last': 0,
                'density_change_from_mean': 0,
                'density_volatility': 0,
            })
    
    return features

def extract_single_exam_features(exam, prefix='current'):
    """
    Extract features from a single mammography exam
    """
    features = {}
    
    # Basic exam features
    features[f'{prefix}_age_group'] = exam['x_age'] if pd.notna(exam['x_age']) else 0
    features[f'{prefix}_exam_year'] = exam['exam_year']
    
    # Mammography features
    if pd.notna(exam['libra_percentdensity']):
        features[f'{prefix}_density'] = exam['libra_percentdensity']
    else:
        features[f'{prefix}_density'] = 0
    
    if pd.notna(exam['libra_breastarea']):
        features[f'{prefix}_breast_area'] = exam['libra_breastarea']
    else:
        features[f'{prefix}_breast_area'] = 0
    
    if pd.notna(exam['libra_densearea']):
        features[f'{prefix}_dense_area'] = exam['libra_densearea']
    else:
        features[f'{prefix}_dense_area'] = 0
    
    # Radiological features
    rad_features = ['rad_timing', 'rad_r1', 'rad_r2', 'rad_recall']
    for feature in rad_features:
        if feature in exam.index and pd.notna(exam[feature]):
            features[f'{prefix}_{feature}'] = exam[feature]
        else:
            features[f'{prefix}_{feature}'] = 0
    
    # Derived features
    if features[f'{prefix}_breast_area'] > 0:
        features[f'{prefix}_dense_ratio'] = features[f'{prefix}_dense_area'] / features[f'{prefix}_breast_area']
    else:
        features[f'{prefix}_dense_ratio'] = 0
    
    return features

def prepare_model_features(features_df):
    """
    Prepare features for model training
    """
    print("\nPreparing features for model training...")
    
    # Select numerical features (exclude IDs and target)
    exclude_cols = ['patient_id', 'current_exam_year', 'cancer_outcome']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"Available features ({len(feature_cols)}): {feature_cols[:10]}..." if len(feature_cols) > 10 else feature_cols)
    
    # Prepare feature matrix
    X = features_df[feature_cols].copy()
    y = features_df['cancer_outcome'].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print(f"Final feature matrix: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

# ===========================================================================================
# MODEL ARCHITECTURE
# ===========================================================================================

class MammographyHistoryModel(nn.Module):
    """Deep learning model for mammography history analysis"""
    
    def __init__(self, num_features, dropout_rate=0.3):
        super(MammographyHistoryModel, self).__init__()
        
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.feature_processor(x).squeeze()

class HistoryDataset(Dataset):
    """Dataset class for mammography history data"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X.values.astype(np.float32))
        self.y = torch.tensor(y.values.astype(np.float32))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===========================================================================================
# TRAINING AND EVALUATION
# ===========================================================================================

def train_model(model, train_loader, val_loader, device, num_epochs=20):
    """Train the model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"\nTraining model for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def evaluate_model(model, test_loader, device):
    """Evaluate the model"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    
    if len(np.unique(y_true)) <= 1:
        return {
            'auc': 0.5, 'auc_ci': [0.5, 0.5],
            'c_index': 0.5, 'c_index_ci': [0.5, 0.5]
        }
    
    try:
        # AUC with confidence interval
        auc, auc_ci = bootstrap_ci(y_true, y_pred, roc_auc_score, n_bootstrap=100)
        
        # C-index with confidence interval  
        c_index, c_index_ci = bootstrap_ci(y_true, y_pred, concordance_index, n_bootstrap=100)
        
        return {
            'auc': auc,
            'auc_ci': auc_ci,
            'c_index': c_index, 
            'c_index_ci': c_index_ci
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'auc': 0.5, 'auc_ci': [0.5, 0.5],
            'c_index': 0.5, 'c_index_ci': [0.5, 0.5]
        }

# ===========================================================================================
# MAIN ANALYSIS PIPELINE
# ===========================================================================================

def run_real_history_analysis():
    """Run the main analysis pipeline using REAL data"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING REAL DATA WITH {config.HISTORY_YEARS} PREVIOUS EXAMS")
    print(f"{'='*80}")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Step 1: Load and analyze real data
    df = load_and_analyze_real_data(config.METADATA_PATH)
    
    # Step 2: Create patient history features
    try:
        features_df = create_patient_history_features(df, config.HISTORY_YEARS)
    except ValueError as e:
        print(f"Error: {e}")
        print("Try reducing the history_years parameter or check data availability.")
        return
    
    # Step 3: Prepare features for modeling
    X, y, feature_cols = prepare_model_features(features_df)
    
    # Step 4: Split data
    print(f"\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_SIZE/(1-config.TEST_SIZE), 
        random_state=42, stratify=y_train
    )
    
    print(f"Data splits:")
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} positive)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} positive)")
    print(f"  Test:  {len(X_test)} samples ({y_test.sum()} positive)")
    
    # Step 5: Create datasets and data loaders
    train_dataset = HistoryDataset(X_train, y_train)
    val_dataset = HistoryDataset(X_val, y_val)
    test_dataset = HistoryDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)
    
    # Step 6: Initialize and train model
    model = MammographyHistoryModel(num_features=len(feature_cols), dropout_rate=config.DROPOUT_RATE)
    trained_model, training_history = train_model(model, train_loader, val_loader, device, config.NUM_EPOCHS)
    
    # Step 7: Evaluate model
    predictions, targets = evaluate_model(trained_model, test_loader, device)
    
    # Step 8: Calculate metrics
    metrics = calculate_metrics(targets, predictions)
    
    # Step 9: Display results
    print(f"\n{'='*80}")
    print(f"RESULTS FOR HISTORY = {config.HISTORY_YEARS} PREVIOUS EXAMS")
    print(f"{'='*80}")
    
    print(f"\nPerformance Metrics:")
    print(f"  AUC (ROC):    {metrics['auc']:.4f} (95% CI: {metrics['auc_ci'][0]:.4f}-{metrics['auc_ci'][1]:.4f})")
    print(f"  C-Index:      {metrics['c_index']:.4f} (95% CI: {metrics['c_index_ci'][0]:.4f}-{metrics['c_index_ci'][1]:.4f})")
    
    print(f"\nDataset Information:")
    print(f"  Total samples: {len(features_df)}")
    print(f"  Features used: {len(feature_cols)}")
    print(f"  History length: {config.HISTORY_YEARS} previous exams")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Step 10: Create visualization
    try:
        if len(np.unique(targets)) > 1:
            plt.figure(figsize=(12, 4))
            
            # Plot 1: ROC Curve
            plt.subplot(1, 3, 1)
            fpr, tpr, _ = roc_curve(targets, predictions)
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {config.HISTORY_YEARS} Previous Exams')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Plot 2: Prediction Distribution
            plt.subplot(1, 3, 2)
            plt.hist(predictions[targets == 0], bins=30, alpha=0.7, label='No Cancer', density=True)
            plt.hist(predictions[targets == 1], bins=30, alpha=0.7, label='Cancer', density=True)
            plt.xlabel('Prediction Probability')
            plt.ylabel('Density')
            plt.title(f'Prediction Distribution')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Plot 3: Training History
            plt.subplot(1, 3, 3)
            plt.plot(training_history['train_losses'], label='Train Loss')
            plt.plot(training_history['val_losses'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'real_history_{config.HISTORY_YEARS}_exams_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nPlot saved as: real_history_{config.HISTORY_YEARS}_exams_results.png")
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Step 11: Save results
    try:
        results_dict = {
            'history_exams': config.HISTORY_YEARS,
            'auc': metrics['auc'],
            'auc_ci_lower': metrics['auc_ci'][0],
            'auc_ci_upper': metrics['auc_ci'][1],
            'c_index': metrics['c_index'],
            'c_index_ci_lower': metrics['c_index_ci'][0],
            'c_index_ci_upper': metrics['c_index_ci'][1],
            'n_samples': len(features_df),
            'n_features': len(feature_cols),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_df = pd.DataFrame([results_dict])
        results_df.to_csv(f'real_history_{config.HISTORY_YEARS}_exams_metrics.csv', index=False)
        print(f"Results saved as: real_history_{config.HISTORY_YEARS}_exams_metrics.csv")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print(f"\n{'='*80}")
    print("REAL DATA ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return metrics

# ===========================================================================================
# EXECUTION
# ===========================================================================================

if __name__ == "__main__":
    print(f"Starting REAL data analysis with HISTORY = {HISTORY} previous exams")
    results = run_real_history_analysis()
    print(f"\nFinal Results Summary:")
    print(f"History: {HISTORY} previous exams")
    print(f"AUC: {results['auc']:.4f} (CI: {results['auc_ci'][0]:.4f}-{results['auc_ci'][1]:.4f})")
    print(f"C-Index: {results['c_index']:.4f} (CI: {results['c_index_ci'][0]:.4f}-{results['c_index_ci'][1]:.4f})")
