# ===========================================================================================
# BREAST CANCER RISK PREDICTION USING TRANSFORMER MODEL
# ===========================================================================================
# This script is designed for Kaggle notebooks with 2x T4 GPUs
# Run each section in separate Kaggle cells as indicated by comments

# ===========================================================================================
# CELL 1: Install Required Packages and Import Libraries
# ===========================================================================================

# Install required packages (run this in first cell)
!pip install transformers torch torchvision pydicom scikit-learn pandas numpy matplotlib seaborn
!pip install timm accelerate datasets
!pip install opencv-python-headless pillow

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import timm
import pydicom
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ===========================================================================================
# CELL 2: Data Loading and Initial Exploration
# ===========================================================================================

# Load the metadata
metadata_path = "/kaggle/input/breast-cancer-research-metadata/CSAW-CC_breast_cancer_screening_data.csv"
df = pd.read_csv(metadata_path)

print("Metadata shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Check unique values for categorical columns
categorical_columns = ['x_case', 'x_cancer_laterality', 'x_type', 'x_lymphnode_met', 
                      'rad_recall', 'rad_recall_type_right', 'rad_recall_type_left', 
                      'imagelaterality', 'viewposition']

for col in categorical_columns:
    if col in df.columns:
        print(f"\n{col} unique values: {df[col].unique()}")

# ===========================================================================================
# CELL 3: Data Preprocessing - Metadata Cleaning
# ===========================================================================================

def preprocess_metadata(df):
    """Preprocess the metadata for model training"""
    df_processed = df.copy()
    
    # Handle missing values
    print("Handling missing values...")
    
    # Fill missing numerical values with median
    numerical_cols = ['x_age', 'libra_breastarea', 'libra_densearea', 'libra_percentdensity']
    for col in numerical_cols:
        if col in df_processed.columns:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
    
    # Fill missing categorical values with mode or 'Unknown'
    for col in categorical_columns:
        if col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna('Unknown')
            else:
                mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 0
                df_processed[col] = df_processed[col].fillna(mode_val)
    
    # Create binary cancer outcome variable
    # Assuming x_case indicates cancer case (1) or control (0)
    df_processed['cancer_outcome'] = df_processed['x_case'].apply(lambda x: 1 if x == 1 else 0)
    
    # Create time-to-event variables (simulated based on exam_year for demonstration)
    # In real scenario, you would have actual follow-up data
    current_year = 2023  # Adjust based on your dataset
    df_processed['follow_up_years'] = current_year - df_processed['exam_year']
    
    # Create time-dependent outcomes (1-year, 2-year, 3-year, 4-year)
    for years in [1, 2, 3, 4]:
        df_processed[f'cancer_{years}year'] = df_processed.apply(
            lambda row: 1 if (row['cancer_outcome'] == 1 and row['follow_up_years'] <= years) else 0, 
            axis=1
        )
    
    # Encode categorical variables
    le_dict = {}
    for col in categorical_columns:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
    
    return df_processed, le_dict

# Preprocess the data
df_processed, label_encoders = preprocess_metadata(df)
print("Data preprocessing completed!")
print(f"Processed data shape: {df_processed.shape}")

# Check the distribution of outcomes
for years in [1, 2, 3, 4]:
    outcome_col = f'cancer_{years}year'
    if outcome_col in df_processed.columns:
        print(f"\n{outcome_col} distribution:")
        print(df_processed[outcome_col].value_counts())

# ===========================================================================================
# CELL 4: DICOM Image Loading and Preprocessing Functions
# ===========================================================================================

def load_dicom_image(file_path):
    """Load and preprocess DICOM image"""
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(file_path)
        
        # Get pixel array
        image = dicom.pixel_array
        
        # Normalize to 0-255
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Convert to RGB (duplicate grayscale to 3 channels)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        return image
    except Exception as e:
        print(f"Error loading DICOM {file_path}: {str(e)}")
        return None

def get_image_paths(df, base_path):
    """Get full paths to DICOM images"""
    image_paths = []
    missing_files = []
    
    for idx, row in df.iterrows():
        filename = row['anon_filename']
        
        # Check in both batch directories
        path1 = os.path.join(base_path, "Batch_1", "Batch_1", filename)
        path2 = os.path.join(base_path, "Batch_2", "Batch_2", filename)
        
        if os.path.exists(path1):
            image_paths.append(path1)
        elif os.path.exists(path2):
            image_paths.append(path2)
        else:
            image_paths.append(None)
            missing_files.append(filename)
    
    print(f"Found images: {len([p for p in image_paths if p is not None])}")
    print(f"Missing images: {len(missing_files)}")
    
    return image_paths, missing_files

# Get image paths
base_image_path = "/kaggle/input/breast-cancer-research-dataset-batch-1-and-batch-2"
image_paths, missing_files = get_image_paths(df_processed, base_image_path)
df_processed['image_path'] = image_paths

# Filter out rows with missing images
df_processed = df_processed[df_processed['image_path'].notna()].reset_index(drop=True)
print(f"Dataset after filtering missing images: {df_processed.shape}")

# ===========================================================================================
# CELL 5: Custom Dataset Class for DICOM Images
# ===========================================================================================

class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, target_columns, transform=None, include_metadata=True):
        self.df = dataframe.reset_index(drop=True)
        self.target_columns = target_columns
        self.transform = transform
        self.include_metadata = include_metadata
        
        # Prepare metadata features
        if include_metadata:
            self.metadata_features = [
                'x_age', 'libra_breastarea', 'libra_densearea', 'libra_percentdensity',
                'x_cancer_laterality_encoded', 'x_type_encoded', 'x_lymphnode_met_encoded',
                'rad_recall_encoded', 'imagelaterality_encoded', 'viewposition_encoded'
            ]
            # Filter features that exist in the dataframe
            self.metadata_features = [f for f in self.metadata_features if f in self.df.columns]
            
            # Standardize metadata features
            self.scaler = StandardScaler()
            self.metadata_scaled = self.scaler.fit_transform(self.df[self.metadata_features])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess image
        image = load_dicom_image(row['image_path'])
        if image is None:
            # Return a blank image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Resize image
        image = cv2.resize(image, (224, 224))
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Get targets
        targets = {}
        for col in self.target_columns:
            if col in self.df.columns:
                targets[col] = torch.tensor(row[col], dtype=torch.float32)
        
        # Get metadata features
        metadata = None
        if self.include_metadata:
            metadata = torch.tensor(self.metadata_scaled[idx], dtype=torch.float32)
        
        return {
            'image': image,
            'metadata': metadata,
            'targets': targets,
            'patient_id': row['anon_patientid']
        }

# Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Dataset class defined successfully!")

# ===========================================================================================
# CELL 6: Model Architecture - Multimodal Transformer
# ===========================================================================================

class MultimodalBreastCancerModel(nn.Module):
    def __init__(self, num_metadata_features=0, num_classes=4, use_image=True, use_metadata=True):
        super(MultimodalBreastCancerModel, self).__init__()
        
        self.use_image = use_image
        self.use_metadata = use_metadata
        
        # Image encoder (Vision Transformer)
        if use_image:
            self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            image_features = self.image_encoder.num_features  # 768 for ViT-Base
        else:
            image_features = 0
        
        # Metadata encoder
        if use_metadata and num_metadata_features > 0:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(num_metadata_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            metadata_features = 64
        else:
            metadata_features = 0
        
        # Fusion and classification layers
        total_features = image_features + metadata_features
        
        if total_features == 0:
            raise ValueError("At least one modality (image or metadata) must be used")
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Separate heads for each time horizon
        self.classifiers = nn.ModuleDict({
            'cancer_1year': nn.Linear(256, 1),
            'cancer_2year': nn.Linear(256, 1),
            'cancer_3year': nn.Linear(256, 1),
            'cancer_4year': nn.Linear(256, 1)
        })
        
    def forward(self, image=None, metadata=None):
        features = []
        
        # Process image
        if self.use_image and image is not None:
            image_feat = self.image_encoder(image)
            features.append(image_feat)
        
        # Process metadata
        if self.use_metadata and metadata is not None:
            metadata_feat = self.metadata_encoder(metadata)
            features.append(metadata_feat)
        
        # Concatenate features
        if len(features) > 1:
            combined_features = torch.cat(features, dim=1)
        else:
            combined_features = features[0]
        
        # Fusion
        fused_features = self.fusion(combined_features)
        
        # Classification for each time horizon
        outputs = {}
        for time_horizon, classifier in self.classifiers.items():
            outputs[time_horizon] = torch.sigmoid(classifier(fused_features))
        
        return outputs

print("Model architecture defined successfully!")

# ===========================================================================================
# CELL 7: Training Function
# ===========================================================================================

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda'):
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': {horizon: [] for horizon in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']}
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            image = batch['image'].to(device) if batch['image'] is not None else None
            metadata = batch['metadata'].to(device) if batch['metadata'] is not None else None
            
            outputs = model(image=image, metadata=metadata)
            
            # Calculate loss for each time horizon
            total_loss = 0
            for horizon in outputs.keys():
                if horizon in batch['targets']:
                    target = batch['targets'][horizon].to(device)
                    loss = criterion(outputs[horizon].squeeze(), target)
                    total_loss += loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = {horizon: [] for horizon in outputs.keys()}
        val_targets = {horizon: [] for horizon in outputs.keys()}
        
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device) if batch['image'] is not None else None
                metadata = batch['metadata'].to(device) if batch['metadata'] is not None else None
                
                outputs = model(image=image, metadata=metadata)
                
                # Calculate validation loss and collect predictions
                batch_val_loss = 0
                for horizon in outputs.keys():
                    if horizon in batch['targets']:
                        target = batch['targets'][horizon].to(device)
                        loss = criterion(outputs[horizon].squeeze(), target)
                        batch_val_loss += loss
                        
                        val_predictions[horizon].extend(outputs[horizon].squeeze().cpu().numpy())
                        val_targets[horizon].extend(target.cpu().numpy())
                
                val_loss += batch_val_loss.item()
        
        # Calculate AUC scores
        val_aucs = {}
        for horizon in val_predictions.keys():
            if len(set(val_targets[horizon])) > 1:  # Check if we have both classes
                auc = roc_auc_score(val_targets[horizon], val_predictions[horizon])
                val_aucs[horizon] = auc
                history['val_auc'][horizon].append(auc)
            else:
                val_aucs[horizon] = 0.0
                history['val_auc'][horizon].append(0.0)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            for horizon, auc in val_aucs.items():
                print(f"{horizon} AUC: {auc:.4f}")
            print("-" * 50)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

print("Training function defined successfully!")

# ===========================================================================================
# CELL 8: Evaluation Function
# ===========================================================================================

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and return predictions and metrics"""
    model.eval()
    model.to(device)
    
    predictions = {horizon: [] for horizon in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']}
    targets = {horizon: [] for horizon in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']}
    patient_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            image = batch['image'].to(device) if batch['image'] is not None else None
            metadata = batch['metadata'].to(device) if batch['metadata'] is not None else None
            
            outputs = model(image=image, metadata=metadata)
            
            # Collect predictions and targets
            for horizon in outputs.keys():
                if horizon in batch['targets']:
                    predictions[horizon].extend(outputs[horizon].squeeze().cpu().numpy())
                    targets[horizon].extend(batch['targets'][horizon].cpu().numpy())
            
            patient_ids.extend(batch['patient_id'])
    
    # Calculate metrics
    metrics = {}
    for horizon in predictions.keys():
        if len(set(targets[horizon])) > 1:  # Check if we have both classes
            auc = roc_auc_score(targets[horizon], predictions[horizon])
            
            # Calculate confidence intervals (bootstrap)
            n_bootstrap = 1000
            rng = np.random.RandomState(42)
            bootstrap_aucs = []
            
            for _ in range(n_bootstrap):
                indices = rng.choice(len(targets[horizon]), len(targets[horizon]), replace=True)
                if len(set([targets[horizon][i] for i in indices])) > 1:
                    bootstrap_auc = roc_auc_score(
                        [targets[horizon][i] for i in indices],
                        [predictions[horizon][i] for i in indices]
                    )
                    bootstrap_aucs.append(bootstrap_auc)
            
            # Calculate 95% confidence interval
            ci_lower = np.percentile(bootstrap_aucs, 2.5)
            ci_upper = np.percentile(bootstrap_aucs, 97.5)
            
            metrics[horizon] = {
                'auc': auc,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'predictions': predictions[horizon],
                'targets': targets[horizon]
            }
        else:
            metrics[horizon] = {
                'auc': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'predictions': predictions[horizon],
                'targets': targets[horizon]
            }
    
    return metrics, patient_ids

def create_results_table(metrics_dict, model_names):
    """Create results table similar to the research paper format"""
    
    time_horizons = ['1-year', '2-year', '3-year', '4-year']
    
    results_data = []
    
    for model_name in model_names:
        if model_name in metrics_dict:
            row = {'Model': model_name}
            
            for i, horizon_key in enumerate(['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']):
                horizon_name = time_horizons[i]
                
                if horizon_key in metrics_dict[model_name]:
                    auc = metrics_dict[model_name][horizon_key]['auc']
                    ci_lower = metrics_dict[model_name][horizon_key]['ci_lower']
                    ci_upper = metrics_dict[model_name][horizon_key]['ci_upper']
                    
                    # Format like in the paper: 0.XX±0.XX
                    ci_range = (ci_upper - ci_lower) / 2
                    formatted_result = f"{auc:.2f}±{ci_range:.2f}"
                else:
                    formatted_result = "N/A"
                
                row[horizon_name] = formatted_result
            
            results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    return results_df

print("Evaluation function defined successfully!")

# ===========================================================================================
# CELL 9: Data Splitting and Dataset Creation
# ===========================================================================================

# Split data by patient ID to avoid data leakage
unique_patients = df_processed['anon_patientid'].unique()
train_patients, temp_patients = train_test_split(unique_patients, test_size=0.4, random_state=42, 
                                                stratify=None)  # Remove stratify if causing issues
val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=42)

# Create train, validation, and test sets
train_df = df_processed[df_processed['anon_patientid'].isin(train_patients)].reset_index(drop=True)
val_df = df_processed[df_processed['anon_patientid'].isin(val_patients)].reset_index(drop=True)
test_df = df_processed[df_processed['anon_patientid'].isin(test_patients)].reset_index(drop=True)

print(f"Train set: {len(train_df)} samples ({len(train_patients)} patients)")
print(f"Validation set: {len(val_df)} samples ({len(val_patients)} patients)")
print(f"Test set: {len(test_df)} samples ({len(test_patients)} patients)")

# Check outcome distribution in each split
target_columns = ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']

for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
    print(f"\n{split_name} set outcome distribution:")
    for col in target_columns:
        if col in split_df.columns:
            print(f"{col}: {split_df[col].value_counts().to_dict()}")

# Create datasets
num_metadata_features = len([f for f in ['x_age', 'libra_breastarea', 'libra_densearea', 'libra_percentdensity',
                                         'x_cancer_laterality_encoded', 'x_type_encoded', 'x_lymphnode_met_encoded',
                                         'rad_recall_encoded', 'imagelaterality_encoded', 'viewposition_encoded'] 
                            if f in train_df.columns])

print(f"Number of metadata features: {num_metadata_features}")

# ===========================================================================================
# CELL 10: Train and Evaluate Different Model Configurations (Ablation Study)
# ===========================================================================================

# Configuration for different models (ablation study)
model_configs = {
    'Image-Only': {'use_image': True, 'use_metadata': False},
    'Metadata-Only': {'use_image': False, 'use_metadata': True}, 
    'Multimodal': {'use_image': True, 'use_metadata': True}
}

# Storage for results
all_metrics = {}
trained_models = {}

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Batch size (adjust based on GPU memory)
batch_size = 8  # Reduce if you get OOM errors

for model_name, config in model_configs.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name} Model")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = BreastCancerDataset(
        train_df, target_columns, 
        transform=train_transform, 
        include_metadata=config['use_metadata']
    )
    
    val_dataset = BreastCancerDataset(
        val_df, target_columns, 
        transform=val_transform, 
        include_metadata=config['use_metadata']
    )
    
    test_dataset = BreastCancerDataset(
        test_df, target_columns, 
        transform=val_transform, 
        include_metadata=config['use_metadata']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = MultimodalBreastCancerModel(
        num_metadata_features=num_metadata_features,
        use_image=config['use_image'],
        use_metadata=config['use_metadata']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trained_model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=30,  # Reduce for faster training, increase for better results
        learning_rate=1e-4,
        device=device
    )
    
    # Evaluate on test set
    test_metrics, test_patient_ids = evaluate_model(trained_model, test_loader, device)
    
    # Store results
    all_metrics[model_name] = test_metrics
    trained_models[model_name] = {
        'model': trained_model,
        'history': history
    }
    
    # Print results for this model
    print(f"\n{model_name} Test Results:")
    print("-" * 40)
    for horizon, metrics in test_metrics.items():
        auc = metrics['auc']
        ci_lower = metrics['ci_lower'] 
        ci_upper = metrics['ci_upper']
        print(f"{horizon}: AUC = {auc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")

print("\nAll models trained successfully!")

# ===========================================================================================
# CELL 11: Generate Results Table (Similar to Research Paper Format)
# ===========================================================================================

# Create comprehensive results table
model_names = list(all_metrics.keys())
results_table = create_results_table(all_metrics, model_names)

print("FINAL RESULTS - Time-dependent AUC")
print("=" * 60)
print(results_table.to_string(index=False))

# Create a more detailed table with separate confidence intervals
detailed_results = []
for model_name in model_names:
    row = [model_name]
    for horizon_key in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']:
        if horizon_key in all_metrics[model_name]:
            metrics = all_metrics[model_name][horizon_key]
            auc = metrics['auc']
            ci_lower = metrics['ci_lower']
            ci_upper = metrics['ci_upper']
            row.append(f"{auc:.3f}")
            row.append(f"({ci_lower:.3f}-{ci_upper:.3f})")
        else:
            row.extend(["N/A", "N/A"])
    detailed_results.append(row)

# Create detailed DataFrame
columns = ['Model']
for horizon in ['1-year', '2-year', '3-year', '4-year']:
    columns.extend([f'{horizon} AUC', f'{horizon} 95% CI'])

detailed_df = pd.DataFrame(detailed_results, columns=columns)
print("\nDETAILED RESULTS TABLE")
print("=" * 80)
print(detailed_df.to_string(index=False))

# Save results to CSV
results_table.to_csv('breast_cancer_results_summary.csv', index=False)
detailed_df.to_csv('breast_cancer_results_detailed.csv', index=False)
print("\nResults saved to CSV files!")

# ===========================================================================================
# CELL 12: Visualization and Analysis
# ===========================================================================================

# Plot ROC curves for best performing model
best_model_name = max(all_metrics.keys(), 
                     key=lambda x: np.mean([all_metrics[x][h]['auc'] for h in all_metrics[x].keys()]))

print(f"Best performing model: {best_model_name}")

# Create ROC curve plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = ['blue', 'red', 'green', 'orange']
horizons = ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']
horizon_names = ['1-Year', '2-Year', '3-Year', '4-Year']

for i, (horizon_key, horizon_name, color) in enumerate(zip(horizons, horizon_names, colors)):
    if horizon_key in all_metrics[best_model_name]:
        metrics = all_metrics[best_model_name][horizon_key]
        targets = metrics['targets']
        predictions = metrics['predictions']
        
        if len(set(targets)) > 1:
            fpr, tpr, _ = roc_curve(targets, predictions)
            auc = metrics['auc']
            
            axes[i].plot(fpr, tpr, color=color, linewidth=2, 
                        label=f'ROC curve (AUC = {auc:.3f})')
            axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{horizon_name} Risk Prediction')
            axes[i].legend(loc="lower right")
            axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot training history for the best model
if best_model_name in trained_models:
    history = trained_models[best_model_name]['history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Training Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # AUC curves
    for horizon_key, horizon_name, color in zip(horizons, horizon_names, colors):
        if horizon_key in history['val_auc']:
            ax2.plot(history['val_auc'][horizon_key], label=f'{horizon_name}', color=color)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation AUC')
    ax2.set_title('Validation AUC by Time Horizon')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===========================================================================================
# CELL 13: Additional Analysis - Feature Importance for Metadata
# ===========================================================================================

# Analyze metadata-only model performance to understand feature importance
if 'Metadata-Only' in trained_models:
    print("Metadata Feature Analysis")
    print("=" * 40)
    
    # Get metadata features used
    metadata_features = [f for f in ['x_age', 'libra_breastarea', 'libra_densearea', 'libra_percentdensity',
                                    'x_cancer_laterality_encoded', 'x_type_encoded', 'x_lymphnode_met_encoded',
                                    'rad_recall_encoded', 'imagelaterality_encoded', 'viewposition_encoded'] 
                        if f in test_df.columns]
    
    print(f"Metadata features used: {metadata_features}")
    
    # Train a simple random forest for feature importance analysis
    rf_results = {}
    
    for horizon_key in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']:
        if horizon_key in test_df.columns:
            X = test_df[metadata_features].fillna(0)
            y = test_df[horizon_key]
            
            if len(set(y)) > 1:  # Only if we have both classes
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Get feature importance
                importance = rf.feature_importances_
                feature_importance = dict(zip(metadata_features, importance))
                
                rf_results[horizon_key] = {
                    'feature_importance': feature_importance,
                    'rf_auc': roc_auc_score(y, rf.predict_proba(X)[:, 1])
                }
    
    # Display feature importance
    for horizon_key, results in rf_results.items():
        print(f"\n{horizon_key} - Feature Importance:")
        sorted_features = sorted(results['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.3f}")
        print(f"  Random Forest AUC: {results['rf_auc']:.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nSummary of Results:")
print(f"- Trained {len(model_configs)} different model configurations")
print(f"- Best performing model: {best_model_name}")
print(f"- Results saved to CSV files for further analysis")
print(f"- ROC curves and training history plots generated")
print("\nYou can now use these results to create tables similar to your research paper!")

# ===========================================================================================
# END OF SCRIPT
# ===========================================================================================
