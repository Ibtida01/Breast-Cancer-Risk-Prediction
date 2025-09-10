#!/usr/bin/env python3
"""
MAMMOGRAPHY HISTORY ANALYSIS: IMPACT OF HISTORICAL DATA ON PREDICTION
=====================================================================
Research Question: How does the amount of patient mammography history affect 
breast cancer prediction performance?

This script trains models with different amounts of patient history:
- history = 0: Current mammogram only
- history = 1: Current + 1 year back (2 years total)
- history = 2: Current + 2 years back (3 years total)
- history = 3: Current + 3 years back (4 years total)
- history = 4: Current + 4 years back (5 years total)

Usage: Set the HISTORY variable and run the script to get C-Index and ROC AUC results.

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
import torchvision.transforms as transforms
import timm
import pydicom
from PIL import Image
import cv2
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
    # Dataset paths (adjust these to your Kaggle dataset paths)
    METADATA_PATH = "/kaggle/input/breast-cancer-research-metadata/CSAW-CC_breast_cancer_screening_data.csv"
    IMAGES_PATH = "/kaggle/input/breast-cancer-research-dataset-batch-1-and-batch-2"
    
    # History configuration
    HISTORY_YEARS = HISTORY  # Number of historical years to include (0-4)
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224
    
    # Model parameters  
    DROPOUT_RATE = 0.3
    
    # Data split parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    # Evaluation parameters
    N_BOOTSTRAP = 1000
    K_FOLD = 5

config = Config()

print("="*80)
print("MAMMOGRAPHY HISTORY ANALYSIS")
print("="*80)
print(f"HISTORY SETTING: {HISTORY} years")
print(f"- Training data: Current + {HISTORY} previous years = {HISTORY + 1} total years")
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
        
        for _ in range(n_bootstrap):
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
# DATA PREPROCESSING
# ===========================================================================================

def preprocess_patient_data(metadata_path, history_years=0):
    """
    Preprocess data based on patient ID and create historical datasets
    """
    print(f"\nLoading and preprocessing data for {history_years} years of history...")
    
    # Load metadata
    try:
        df = pd.read_csv(metadata_path, low_memory=False)
        print(f"Original dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Dataset file not found. Creating synthetic data for demonstration...")
        df = create_synthetic_patient_data()
    
    # Clean data
    df_clean = df.dropna(subset=['anon_filename']) if 'anon_filename' in df.columns else df
    if 'anon_filename' in df_clean.columns:
        df_clean = df_clean[df_clean['anon_filename'].str.contains('.dcm', na=False)]
    print(f"After cleaning: {df_clean.shape}")
    
    # Create synthetic patient IDs if not available
    if 'patient_id' not in df_clean.columns:
        print("Creating synthetic patient IDs...")
        # Group similar patients together
        if 'age_at_study' in df_clean.columns:
            # Group by age ranges to simulate patients
            df_clean['age_group'] = (df_clean['age_at_study'] // 5) * 5  # 5-year age groups
            df_clean['patient_id'] = df_clean.groupby(['age_group']).ngroup()
        else:
            # Random patient IDs
            np.random.seed(42)
            n_patients = len(df_clean) // 3  # Assume each patient has ~3 visits on average
            df_clean['patient_id'] = np.random.randint(0, n_patients, len(df_clean))
    
    # Create visit years if not available
    if 'visit_year' not in df_clean.columns:
        print("Creating synthetic visit years...")
        # Assign random visit years to each patient
        np.random.seed(42)
        patient_visits = {}
        for idx, row in df_clean.iterrows():
            pid = row['patient_id']
            if pid not in patient_visits:
                # Each patient gets visits in years 0, -1, -2, -3, -4 (some missing randomly)
                possible_years = list(range(-history_years, 1))  # From -history_years to 0
                n_visits = np.random.randint(1, len(possible_years) + 1)
                patient_visits[pid] = sorted(np.random.choice(possible_years, n_visits, replace=False), reverse=True)
        
        # Assign visit years
        visit_assignments = []
        for idx, row in df_clean.iterrows():
            pid = row['patient_id']
            visits = patient_visits[pid]
            # Assign this record to one of the patient's visits
            visit_year = np.random.choice(visits)
            visit_assignments.append(visit_year)
        
        df_clean['visit_year'] = visit_assignments
    
    # Ensure we have cancer outcome
    if 'x_case' not in df_clean.columns and 'cancer_outcome' not in df_clean.columns:
        print("Creating synthetic cancer outcomes...")
        np.random.seed(42)
        df_clean['x_case'] = np.random.choice([0, 1], size=len(df_clean), p=[0.9, 0.1])
    
    cancer_col = 'x_case' if 'x_case' in df_clean.columns else 'cancer_outcome'
    
    # Create features for the specified history length
    print(f"Processing patients with up to {history_years} years of history...")
    
    # Group by patient
    patient_groups = df_clean.groupby('patient_id')
    processed_patients = []
    
    for patient_id, patient_data in patient_groups:
        # Sort by visit year (most recent first)
        patient_data = patient_data.sort_values('visit_year', ascending=False)
        
        # Get required number of visits (current + history_years)
        required_visits = history_years + 1
        available_visits = len(patient_data)
        
        if available_visits == 0:
            continue
            
        # Take up to required_visits (pad with last visit if needed)
        if available_visits >= required_visits:
            selected_visits = patient_data.head(required_visits)
        else:
            # Pad with duplicate of most recent visit
            selected_visits = patient_data.copy()
            last_visit = patient_data.iloc[0:1]
            for i in range(required_visits - available_visits):
                padded_visit = last_visit.copy()
                padded_visit['visit_year'] = -i-1  # Assign historical visit years
                selected_visits = pd.concat([selected_visits, padded_visit], ignore_index=True)
        
        # Sort by visit year (most recent first)
        selected_visits = selected_visits.sort_values('visit_year', ascending=False)
        
        # Create aggregated features for this patient
        current_visit = selected_visits.iloc[0]  # Most recent
        
        # Aggregate features from all visits
        aggregated_features = {
            'patient_id': patient_id,
            'anon_filename': current_visit.get('anon_filename', f'patient_{patient_id}_current.dcm'),
            'cancer_outcome': current_visit[cancer_col],
            'history_length': len(selected_visits),
            'visit_years': ','.join(map(str, selected_visits['visit_year'].tolist()))
        }
        
        # Add age features
        if 'age_at_study' in selected_visits.columns:
            ages = selected_visits['age_at_study'].dropna()
            if len(ages) > 0:
                aggregated_features.update({
                    'current_age': ages.iloc[0],
                    'mean_age': ages.mean(),
                    'age_trend': ages.iloc[0] - ages.iloc[-1] if len(ages) > 1 else 0
                })
        
        # Add other clinical features if available
        clinical_features = ['menopause_status', 'hrt_status', 'birads_density', 
                           'family_history_breast', 'family_history_ovarian',
                           'previous_benign_biopsy', 'previous_cancer']
        
        for feature in clinical_features:
            if feature in selected_visits.columns:
                values = selected_visits[feature].dropna()
                if len(values) > 0:
                    if feature in ['birads_density']:  # Numerical
                        aggregated_features[f'{feature}_current'] = values.iloc[0]
                        aggregated_features[f'{feature}_mean'] = values.mean()
                    else:  # Categorical
                        aggregated_features[f'{feature}_current'] = values.iloc[0]
                        aggregated_features[f'{feature}_mode'] = values.mode().iloc[0] if len(values.mode()) > 0 else values.iloc[0]
        
        processed_patients.append(aggregated_features)
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_patients)
    print(f"Processed {len(processed_df)} patients with {history_years} years of history")
    print(f"Cancer cases: {processed_df['cancer_outcome'].sum()}/{len(processed_df)} ({100*processed_df['cancer_outcome'].mean():.1f}%)")
    
    return processed_df

def create_synthetic_patient_data():
    """Create synthetic patient data for testing"""
    print("Creating synthetic patient data...")
    
    np.random.seed(42)
    n_records = 1000
    n_patients = 300
    
    data = []
    for i in range(n_records):
        patient_id = np.random.randint(0, n_patients)
        visit_year = np.random.choice([-4, -3, -2, -1, 0], p=[0.1, 0.15, 0.2, 0.25, 0.3])
        
        record = {
            'anon_filename': f'patient_{patient_id}_year_{visit_year}.dcm',
            'patient_id': patient_id,
            'visit_year': visit_year,
            'age_at_study': np.random.randint(40, 80),
            'x_case': np.random.choice([0, 1], p=[0.9, 0.1]),
            'menopause_status': np.random.choice(['pre', 'post']),
            'hrt_status': np.random.choice(['never', 'current', 'former']),
            'birads_density': np.random.randint(1, 5),
            'family_history_breast': np.random.choice([0, 1], p=[0.85, 0.15])
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    print(f"Created synthetic dataset: {len(df)} records for {n_patients} patients")
    return df

def prepare_features(df):
    """Prepare features for model training"""
    print("Preparing features for model training...")
    
    # Select numerical features
    numerical_features = []
    for col in df.columns:
        if col.endswith(('_current', '_mean', '_trend')) and df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
    
    # Add basic features if available
    basic_features = ['current_age', 'mean_age', 'age_trend', 'history_length']
    for feat in basic_features:
        if feat in df.columns:
            numerical_features.append(feat)
    
    # Select categorical features
    categorical_features = []
    for col in df.columns:
        if col.endswith(('_current', '_mode')) and df[col].dtype == 'object':
            categorical_features.append(col)
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Handle missing values
    feature_df = df[numerical_features + categorical_features + ['cancer_outcome', 'anon_filename', 'patient_id']].copy()
    
    # Fill numerical missing values
    for col in numerical_features:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    
    # Encode categorical features
    le_dict = {}
    for col in categorical_features:
        feature_df[col] = feature_df[col].fillna('Unknown').astype(str)
        le = LabelEncoder()
        feature_df[f'{col}_encoded'] = le.fit_transform(feature_df[col])
        le_dict[col] = le
        categorical_features.append(f'{col}_encoded')
    
    # Final feature list
    final_features = numerical_features + [f'{col}_encoded' for col in categorical_features if not col.endswith('_encoded')]
    final_features = [col for col in final_features if col in feature_df.columns]
    
    print(f"Final features ({len(final_features)}): {final_features}")
    
    return feature_df, final_features, le_dict

# ===========================================================================================
# MODEL ARCHITECTURE
# ===========================================================================================

class HistoryAwareModel(nn.Module):
    """Model that processes patient data with historical context"""
    
    def __init__(self, num_features, dropout_rate=0.3):
        super(HistoryAwareModel, self).__init__()
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final prediction layer
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_processor(x)
        output = self.classifier(features)
        return output

class HistoryDataset(Dataset):
    """Dataset class for patient data with history"""
    
    def __init__(self, df, feature_columns, target_column):
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Prepare features and targets
        self.features = self.df[feature_columns].values.astype(np.float32)
        self.targets = self.df[target_column].values.astype(np.float32)
        
        # Handle any remaining NaN values
        self.features = np.nan_to_num(self.features, nan=0.0)
        
        print(f"Dataset created: {len(self.features)} samples, {len(feature_columns)} features")
        print(f"Positive cases: {self.targets.sum()}/{len(self.targets)} ({100*self.targets.mean():.1f}%)")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

# ===========================================================================================
# TRAINING AND EVALUATION
# ===========================================================================================

def train_model(model, train_loader, val_loader, device, num_epochs=15):
    """Train the model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
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
        
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
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
    """Evaluate the model and return predictions"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(features).squeeze()
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    print(f"Evaluation completed: {len(predictions)} predictions")
    print(f"Positive cases: {targets.sum()}/{len(targets)} ({100*targets.mean():.1f}%)")
    
    return predictions, targets

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    
    if len(np.unique(y_true)) <= 1:
        print("Warning: Only one class present in targets")
        return {
            'auc': 0.5, 'auc_ci': [0.5, 0.5],
            'c_index': 0.5, 'c_index_ci': [0.5, 0.5]
        }
    
    try:
        # AUC with confidence interval
        auc, auc_ci = bootstrap_ci(y_true, y_pred, roc_auc_score, n_bootstrap=config.N_BOOTSTRAP)
        
        # C-index with confidence interval  
        c_index, c_index_ci = bootstrap_ci(y_true, y_pred, concordance_index, n_bootstrap=config.N_BOOTSTRAP)
        
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

def run_history_analysis():
    """Run the main analysis pipeline"""
    
    print(f"\n{'='*80}")
    print(f"RUNNING ANALYSIS WITH {config.HISTORY_YEARS} YEARS OF HISTORY")
    print(f"{'='*80}")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Step 1: Preprocess data based on patient ID and history
    processed_df = preprocess_patient_data(config.METADATA_PATH, config.HISTORY_YEARS)
    
    if len(processed_df) == 0:
        print("No data available for processing")
        return
    
    # Step 2: Prepare features
    feature_df, feature_columns, le_dict = prepare_features(processed_df)
    
    if len(feature_columns) == 0:
        print("No features available for training")
        return
    
    # Step 3: Split data
    print(f"\nSplitting data...")
    train_df, test_df = train_test_split(
        feature_df, 
        test_size=config.TEST_SIZE, 
        random_state=42,
        stratify=feature_df['cancer_outcome']
    )
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=config.VAL_SIZE/(1-config.TEST_SIZE),
        random_state=42,
        stratify=train_df['cancer_outcome']
    )
    
    print(f"Data splits:")
    print(f"  Train: {len(train_df)} samples ({train_df['cancer_outcome'].sum()} positive)")
    print(f"  Val:   {len(val_df)} samples ({val_df['cancer_outcome'].sum()} positive)")
    print(f"  Test:  {len(test_df)} samples ({test_df['cancer_outcome'].sum()} positive)")
    
    # Step 4: Create datasets and data loaders
    train_dataset = HistoryDataset(train_df, feature_columns, 'cancer_outcome')
    val_dataset = HistoryDataset(val_df, feature_columns, 'cancer_outcome')
    test_dataset = HistoryDataset(test_df, feature_columns, 'cancer_outcome')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)
    
    # Step 5: Initialize and train model
    model = HistoryAwareModel(num_features=len(feature_columns), dropout_rate=config.DROPOUT_RATE)
    trained_model, training_history = train_model(model, train_loader, val_loader, device, config.NUM_EPOCHS)
    
    # Step 6: Evaluate model
    predictions, targets = evaluate_model(trained_model, test_loader, device)
    
    # Step 7: Calculate metrics
    metrics = calculate_metrics(targets, predictions)
    
    # Step 8: Display results
    print(f"\n{'='*80}")
    print(f"RESULTS FOR HISTORY = {config.HISTORY_YEARS} YEARS")
    print(f"{'='*80}")
    
    print(f"\nPerformance Metrics:")
    print(f"  AUC (ROC):    {metrics['auc']:.4f} (95% CI: {metrics['auc_ci'][0]:.4f}-{metrics['auc_ci'][1]:.4f})")
    print(f"  C-Index:      {metrics['c_index']:.4f} (95% CI: {metrics['c_index_ci'][0]:.4f}-{metrics['c_index_ci'][1]:.4f})")
    
    print(f"\nDataset Information:")
    print(f"  Total patients: {len(processed_df)}")
    print(f"  Features used: {len(feature_columns)}")
    print(f"  History length: {config.HISTORY_YEARS} years")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Step 9: Create simple visualization
    try:
        if len(np.unique(targets)) > 1:
            plt.figure(figsize=(10, 4))
            
            # Plot 1: ROC Curve
            plt.subplot(1, 2, 1)
            fpr, tpr, _ = roc_curve(targets, predictions)
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - History {config.HISTORY_YEARS} Years')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Plot 2: Prediction Distribution
            plt.subplot(1, 2, 2)
            plt.hist(predictions[targets == 0], bins=20, alpha=0.7, label='Negative', density=True)
            plt.hist(predictions[targets == 1], bins=20, alpha=0.7, label='Positive', density=True)
            plt.xlabel('Prediction Probability')
            plt.ylabel('Density')
            plt.title(f'Prediction Distribution - History {config.HISTORY_YEARS} Years')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'history_{config.HISTORY_YEARS}_years_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nPlot saved as: history_{config.HISTORY_YEARS}_years_results.png")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Step 10: Save results
    try:
        results_dict = {
            'history_years': config.HISTORY_YEARS,
            'auc': metrics['auc'],
            'auc_ci_lower': metrics['auc_ci'][0],
            'auc_ci_upper': metrics['auc_ci'][1],
            'c_index': metrics['c_index'],
            'c_index_ci_lower': metrics['c_index_ci'][0],
            'c_index_ci_upper': metrics['c_index_ci'][1],
            'n_patients': len(processed_df),
            'n_features': len(feature_columns),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_df = pd.DataFrame([results_dict])
        results_df.to_csv(f'history_{config.HISTORY_YEARS}_years_metrics.csv', index=False)
        print(f"Results saved as: history_{config.HISTORY_YEARS}_years_metrics.csv")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return metrics

# ===========================================================================================
# EXECUTION
# ===========================================================================================

if __name__ == "__main__":
    print(f"Starting analysis with HISTORY = {HISTORY}")
    results = run_history_analysis()
    print(f"\nFinal Results Summary:")
    print(f"History: {HISTORY} years")
    print(f"AUC: {results['auc']:.4f} (CI: {results['auc_ci'][0]:.4f}-{results['auc_ci'][1]:.4f})")
    print(f"C-Index: {results['c_index']:.4f} (CI: {results['c_index_ci'][0]:.4f}-{results['c_index_ci'][1]:.4f})")
