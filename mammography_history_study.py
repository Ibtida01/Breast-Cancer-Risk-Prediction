#!/usr/bin/env python3
"""
MAMMOGRAPHY HISTORY STUDY: "IS MORE THE MERRIER?"
================================================
Research Question: Does more mammography history improve breast cancer prediction?

This study compares models with different amounts of mammography history:
- No history (current image only)
- 1-year history (current + 1 previous)
- 2-year history (current + 2 previous)
- 3-year history (current + 3 previous)  
- 4-year history (current + 4 previous)

Performance metrics: C-Index and ROC AUC for comparison

Author: Research Team
Date: September 2025
"""

# ===========================================================================================
# INSTALLATION AND IMPORTS
# ===========================================================================================

import subprocess
import sys

def install_packages():
    packages = [
        'transformers', 'torch', 'torchvision', 'pydicom', 'scikit-learn', 
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'timm', 'accelerate', 
        'datasets', 'opencv-python-headless', 'pillow'
    ]
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uncomment the line below if running for the first time
# install_packages()

# Import libraries
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
import torchvision.transforms as transforms
import timm
import pydicom
from PIL import Image
import cv2
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproduc#!/usr/bin/env python3
"""
MAMMOGRAPHY HISTORY STUDY: "IS MORE THE MERRIER?"
================================================
Research Question: Does more mammography history improve breast cancer prediction?

This study compares models with different amounts of mammography history:
- No history (current image only)
- 1-year history (current + 1 previous)
- 2-year history (current + 2 previous)
- 3-year history (current + 3 previous)  
- 4-year history (current + 4 previous)

Performance metrics: C-Index and ROC AUC for comparison

Author: Research Team
Date: September 2025
"""

# ===========================================================================================
# INSTALLATION AND IMPORTS
# ===========================================================================================

import subprocess
import sys

def install_packages():
    packages = [
        'transformers', 'torch', 'torchvision', 'pydicom', 'scikit-learn', 
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'timm', 'accelerate', 
        'datasets', 'opencv-python-headless', 'pillow'
    ]
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uncomment the line below if running for the first time
# install_packages()

# Import libraries
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

print("="*80)
print("MAMMOGRAPHY HISTORY STUDY: IS MORE THE MERRIER?")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ===========================================================================================
# CONFIGURATION
# ===========================================================================================

class Config:
    """Configuration parameters"""
    # Dataset paths (adjust these to your Kaggle dataset paths)
    METADATA_PATH = "/kaggle/input/breast-cancer-research-metadata/CSAW-CC_breast_cancer_screening_data.csv"
    IMAGES_PATH = "/kaggle/input/breast-cancer-research-dataset-batch-1-and-batch-2"
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 15  # Reduced for multiple experiments
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224
    
    # Model parameters
    NUM_CLASSES = 1  # Binary classification for each time horizon
    DROPOUT_RATE = 0.3
    
    # History study parameters
    MAX_HISTORY_YEARS = 4
    HISTORY_CONFIGS = [0, 1, 2, 3, 4]  # Number of historical years to include
    
    # Other parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    N_BOOTSTRAP = 1000

config = Config()

def concordance_index(targets, predictions):
    """
    Calculate C-index (concordance index) manually.
    For binary classification, C-index is equivalent to AUC.
    """
    try:
        return roc_auc_score(targets, predictions)
    except:
        return 0.5  # Random performance if calculation fails

# ===========================================================================================
# DATA PREPROCESSING WITH HISTORY SIMULATION
# ===========================================================================================

def simulate_patient_history(df, max_years=4):
    """
    Simulate mammography history for each patient.
    In real data, you would have actual historical images.
    For simulation, we create synthetic historical data based on current data.
    """
    print(f"\nSimulating {max_years} years of mammography history...")
    
    # Create patient groups (simulate patients having multiple visits)
    unique_patients = df.groupby(['age_at_study', 'menopause_status']).first().reset_index()
    n_patients = len(unique_patients)
    
    # Assign each original record to a patient
    np.random.seed(42)
    df['patient_id'] = np.random.randint(0, n_patients//3, len(df))  # Some patients have multiple visits
    
    # Create historical records for each patient
    history_data = []
    
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].iloc[0].copy()
        
        # Current visit (year 0)
        current_record = patient_data.copy()
        current_record['visit_year'] = 0
        current_record['years_to_cancer'] = np.random.choice([1, 2, 3, 4, 5]) if current_record['x_case'] == 1 else 999
        history_data.append(current_record)
        
        # Historical visits (years -1, -2, -3, -4)
        for year_back in range(1, max_years + 1):
            historical_record = patient_data.copy()
            historical_record['visit_year'] = -year_back
            historical_record['age_at_study'] = max(18, historical_record['age_at_study'] - year_back)
            
            # Simulate changes in historical data
            # Mammographic density might change over time
            if 'mammo_density' in historical_record.index:
                # Density tends to decrease with age
                density_change = np.random.normal(-0.1 * year_back, 0.05)
                historical_record['mammo_density'] = max(1, historical_record['mammo_density'] + density_change)
            
            # Cancer status - only current visit can be cancer positive for prediction
            historical_record['x_case'] = 0
            historical_record['years_to_cancer'] = historical_record['years_to_cancer'] + year_back
            
            history_data.append(historical_record)
    
    history_df = pd.DataFrame(history_data)
    print(f"Created history dataset: {len(history_df)} records for {len(history_df['patient_id'].unique())} patients")
    print(f"Visit years distribution:\n{history_df['visit_year'].value_counts().sort_index()}")
    
    return history_df

def load_and_preprocess_metadata_with_history(file_path, max_history_years=4):
    """Load and preprocess metadata with mammography history simulation"""
    print("Loading and preprocessing metadata with history...")
    
    # Load base dataset
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Original dataset shape: {df.shape}")
    
    # Clean data
    df_clean = df.dropna(subset=['anon_filename'])
    df_clean = df_clean[df_clean['anon_filename'].str.contains('.dcm', na=False)]
    print(f"After cleaning: {df_clean.shape}")
    
    # Select relevant columns
    feature_cols = [
        'anon_filename', 'age_at_study', 'x_case',
        'menopause_status', 'hrt_status', 'birads_density',
        'family_history_breast', 'family_history_ovarian',
        'previous_benign_biopsy', 'previous_cancer'
    ]
    
    available_cols = [col for col in feature_cols if col in df_clean.columns]
    df_subset = df_clean[available_cols].copy()
    print(f"Selected features: {available_cols}")
    
    # Handle missing values
    categorical_cols = ['menopause_status', 'hrt_status', 'family_history_breast', 
                       'family_history_ovarian', 'previous_benign_biopsy', 'previous_cancer']
    
    for col in categorical_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].fillna('Unknown')
    
    # Simulate patient history
    history_df = simulate_patient_history(df_subset, max_history_years)
    
    # Create target variables for different prediction horizons
    print("\nCreating target variables...")
    for years in [1, 2, 3, 4]:
        # Target: will patient develop cancer within X years?
        history_df[f'cancer_{years}year'] = (
            (history_df['x_case'] == 1) & 
            (history_df['years_to_cancer'] <= years)
        ).astype(int)
        
        print(f"  cancer_{years}year: {history_df[f'cancer_{years}year'].sum()} positive cases")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    le_dict = {}
    for col in categorical_cols:
        if col in history_df.columns:
            le = LabelEncoder()
            history_df[f'{col}_encoded'] = le.fit_transform(history_df[col].astype(str))
            le_dict[col] = le
            print(f"  {col}: encoded to {col}_encoded")
    
    return history_df, le_dict

# ===========================================================================================
# HISTORY-AWARE DATASET CLASS
# ===========================================================================================

class HistoryAwareBreastCancerDataset(Dataset):
    """Dataset class that handles different amounts of mammography history"""
    
    def __init__(self, df, image_paths, target_cols, metadata_cols, history_years=0, transform=None):
        """
        Args:
            df: DataFrame with patient data including history
            image_paths: Dictionary mapping filenames to paths
            target_cols: List of target column names
            metadata_cols: List of metadata column names
            history_years: Number of historical years to include (0-4)
            transform: Image transforms
        """
        self.df = df
        self.image_paths = image_paths
        self.target_cols = target_cols
        self.metadata_cols = metadata_cols
        self.history_years = history_years
        self.transform = transform
        
        # Group by patient and visit year for history access
        self.patient_data = {}
        for idx, row in df.iterrows():
            patient_id = row['patient_id']
            visit_year = row['visit_year']
            
            if patient_id not in self.patient_data:
                self.patient_data[patient_id] = {}
            self.patient_data[patient_id][visit_year] = row
        
        # Create samples list (only current visits for prediction)
        self.samples = []
        for idx, row in df.iterrows():
            if row['visit_year'] == 0:  # Only current visits
                self.samples.append(row)
        
        print(f"Dataset with {history_years} years history: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        current_sample = self.samples[idx]
        patient_id = current_sample['patient_id']
        
        # Collect images and metadata for current + historical visits
        images = []
        metadata_features = []
        
        # Current visit (year 0)
        current_image = self.load_image(current_sample['anon_filename'])
        if current_image is not None:
            images.append(current_image)
        
        current_metadata = [current_sample[col] for col in self.metadata_cols 
                          if not pd.isna(current_sample[col])]
        metadata_features.extend(current_metadata)
        
        # Historical visits (years -1, -2, ..., -history_years)
        for year_back in range(1, self.history_years + 1):
            visit_year = -year_back
            if (patient_id in self.patient_data and 
                visit_year in self.patient_data[patient_id]):
                
                historical_sample = self.patient_data[patient_id][visit_year]
                
                # Load historical image
                historical_image = self.load_image(historical_sample['anon_filename'])
                if historical_image is not None:
                    images.append(historical_image)
                
                # Add historical metadata
                historical_metadata = [historical_sample[col] for col in self.metadata_cols 
                                     if not pd.isna(historical_sample[col])]
                metadata_features.extend(historical_metadata)
        
        # Pad or truncate to fixed size
        max_images = self.history_years + 1
        while len(images) < max_images:
            # Duplicate last image if not enough history
            if images:
                images.append(images[-1])
            else:
                # Create dummy image if no images available
                dummy_image = torch.zeros(3, 224, 224)
                images.append(dummy_image)
        
        images = images[:max_images]  # Truncate if too many
        
        # Stack images (shape: [history_length, 3, 224, 224])
        image_tensor = torch.stack(images)
        
        # Metadata features
        if metadata_features:
            metadata_tensor = torch.tensor(metadata_features, dtype=torch.float32)
        else:
            metadata_tensor = torch.zeros(len(self.metadata_cols) * max_images, dtype=torch.float32)
        
        # Targets
        targets = torch.tensor([current_sample[col] for col in self.target_cols], dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'metadata': metadata_tensor,
            'targets': targets,
            'patient_id': patient_id
        }
    
    def load_image(self, filename):
        """Load and preprocess a single DICOM image"""
        if filename not in self.image_paths or pd.isna(filename):
            return None
        
        try:
            # Load DICOM
            dicom_path = self.image_paths[filename]
            dicom_data = pydicom.dcmread(dicom_path)
            image_array = dicom_data.pixel_array
            
            # Normalize to 0-255
            image_array = ((image_array - image_array.min()) / 
                          (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            
            # Convert to 3-channel
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            
            # Convert to PIL and resize
            pil_image = Image.fromarray(image_array)
            pil_image = pil_image.resize((224, 224))
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(pil_image)
            else:
                image_tensor = transforms.ToTensor()(pil_image)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return torch.zeros(3, 224, 224)

# ===========================================================================================
# HISTORY-AWARE MODEL ARCHITECTURE
# ===========================================================================================

class HistoryAwareTransformer(nn.Module):
    """Transformer model that processes multiple mammograms with temporal awareness"""
    
    def __init__(self, num_metadata_features, history_years=0, num_classes=4):
        super(HistoryAwareTransformer, self).__init__()
        
        self.history_years = history_years
        self.num_images = history_years + 1  # Current + historical
        
        # Vision Transformer for each image
        self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.image_feature_dim = self.image_encoder.num_features
        
        # Remove the classification head
        self.image_encoder.head = nn.Identity()
        
        # Temporal encoding for different visits
        self.temporal_embedding = nn.Embedding(self.num_images, self.image_feature_dim)
        
        # Transformer encoder for temporal fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.image_feature_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Metadata processing
        self.metadata_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        fusion_dim = self.image_feature_dim + 64
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Multi-task prediction heads
        self.prediction_heads = nn.ModuleDict({
            f'cancer_{i}year': nn.Linear(256, 1) for i in [1, 2, 3, 4]
        })
    
    def forward(self, images, metadata):
        batch_size = images.shape[0]
        
        # Process each image through ViT
        # images shape: [batch_size, num_images, 3, 224, 224]
        image_features = []
        for i in range(self.num_images):
            if i < images.shape[1]:
                img_feat = self.image_encoder(images[:, i])  # [batch_size, feature_dim]
            else:
                # Pad with zeros if not enough images
                img_feat = torch.zeros(batch_size, self.image_feature_dim, 
                                     device=images.device, dtype=images.dtype)
            image_features.append(img_feat)
        
        # Stack and add temporal embeddings
        image_features = torch.stack(image_features, dim=1)  # [batch_size, num_images, feature_dim]
        
        # Add temporal position embeddings
        temporal_positions = torch.arange(self.num_images, device=images.device)
        temporal_embeds = self.temporal_embedding(temporal_positions)  # [num_images, feature_dim]
        temporal_embeds = temporal_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_images, feature_dim]
        
        image_features = image_features + temporal_embeds
        
        # Apply temporal transformer
        temporal_features = self.temporal_transformer(image_features)  # [batch_size, num_images, feature_dim]
        
        # Global pooling over time dimension
        pooled_features = temporal_features.mean(dim=1)  # [batch_size, feature_dim]
        
        # Process metadata
        metadata_features = self.metadata_mlp(metadata)  # [batch_size, 64]
        
        # Fusion
        fused_features = torch.cat([pooled_features, metadata_features], dim=1)
        fused_features = self.fusion_layer(fused_features)  # [batch_size, 256]
        
        # Multi-task predictions
        predictions = {}
        for task_name, head in self.prediction_heads.items():
            predictions[task_name] = torch.sigmoid(head(fused_features))
        
        return predictions

# ===========================================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ===========================================================================================

def train_history_aware_model(model, train_loader, val_loader, device, num_epochs=15):
    """Train the history-aware model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\nTraining history-aware model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            try:
                images = batch['image'].to(device, non_blocking=True)
                metadata = batch['metadata'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(images, metadata)
                
                # Calculate multi-task loss
                total_loss = 0
                for i, task_name in enumerate(predictions.keys()):
                    task_pred = predictions[task_name].squeeze()
                    task_target = targets[:, i]
                    task_loss = criterion(task_pred, task_target)
                    total_loss += task_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
                
                if batch_idx == 0:  # First batch of epoch
                    print(f"  Epoch {epoch+1}/{num_epochs}, Batch 1 - Loss: {total_loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                try:
                    images = batch['image'].to(device, non_blocking=True)
                    metadata = batch['metadata'].to(device, non_blocking=True)
                    targets = batch['targets'].to(device, non_blocking=True)
                    
                    predictions = model(images, metadata)
                    
                    total_loss = 0
                    for i, task_name in enumerate(predictions.keys()):
                        task_pred = predictions[task_name].squeeze()
                        task_target = targets[:, i]
                        task_loss = criterion(task_pred, task_target)
                        total_loss += task_loss
                    
                    val_loss += total_loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def evaluate_history_model(model, test_loader, device):
    """Evaluate the history-aware model"""
    
    model.eval()
    all_predictions = {f'cancer_{i}year': [] for i in [1, 2, 3, 4]}
    all_targets = {f'cancer_{i}year': [] for i in [1, 2, 3, 4]}
    patient_ids = []
    
    print("Evaluating history-aware model...")
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
                
            try:
                images = batch['image'].to(device, non_blocking=True)
                metadata = batch['metadata'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                batch_patient_ids = batch['patient_id']
                
                predictions = model(images, metadata)
                
                # Store predictions and targets
                for i, task_name in enumerate(predictions.keys()):
                    pred = predictions[task_name].squeeze().cpu().numpy()
                    target = targets[:, i].cpu().numpy()
                    
                    if pred.ndim == 0:
                        pred = pred.reshape(1)
                        target = target.reshape(1)
                    
                    all_predictions[task_name].extend(pred)
                    all_targets[task_name].extend(target)
                
                patient_ids.extend(batch_patient_ids)
                
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    # Calculate metrics
    metrics = {}
    for task_name in all_predictions.keys():
        if len(all_predictions[task_name]) > 0 and len(all_targets[task_name]) > 0:
            preds = np.array(all_predictions[task_name])
            targets = np.array(all_targets[task_name])
            
            # Skip if no positive cases
            if len(np.unique(targets)) > 1:
                auc = roc_auc_score(targets, preds)
                
                # Calculate C-index (same as AUC for binary classification)
                c_index = concordance_index(targets, preds)
                
                # Bootstrap confidence intervals
                aucs = []
                c_indices = []
                for _ in range(100):  # Reduced for speed
                    indices = np.random.choice(len(targets), len(targets), replace=True)
                    if len(np.unique(targets[indices])) > 1:
                        bootstrap_auc = roc_auc_score(targets[indices], preds[indices])
                        bootstrap_c_index = concordance_index(targets[indices], preds[indices])
                        aucs.append(bootstrap_auc)
                        c_indices.append(bootstrap_c_index)
                
                auc_ci = np.percentile(aucs, [2.5, 97.5]) if aucs else [auc, auc]
                c_index_ci = np.percentile(c_indices, [2.5, 97.5]) if c_indices else [c_index, c_index]
                
                metrics[task_name] = {
                    'auc': auc,
                    'auc_ci_lower': auc_ci[0],
                    'auc_ci_upper': auc_ci[1],
                    'c_index': c_index,
                    'c_index_ci_lower': c_index_ci[0],
                    'c_index_ci_upper': c_index_ci[1],
                    'predictions': preds,
                    'targets': targets
                }
    
    return metrics, patient_ids

# ===========================================================================================
# MAIN HISTORY STUDY PIPELINE
# ===========================================================================================

def run_history_study():
    """Run the complete mammography history study"""
    
    print("="*80)
    print("STARTING MAMMOGRAPHY HISTORY STUDY")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Load data with history simulation
    try:
        df_history, le_dict = load_and_preprocess_metadata_with_history(
            config.METADATA_PATH, 
            max_history_years=config.MAX_HISTORY_YEARS
        )
    except FileNotFoundError:
        print("Dataset not found. Creating synthetic dataset for demonstration...")
        # Create synthetic data for testing
        df_history = create_synthetic_history_data()
        le_dict = {}
    
    # Get image paths (simulated)
    print("\nSetting up image paths...")
    image_paths = {}
    unique_filenames = df_history['anon_filename'].unique()
    for filename in unique_filenames:
        # In real implementation, map to actual DICOM files
        image_paths[filename] = f"synthetic_path/{filename}"
    
    # Prepare features
    target_cols = ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']
    metadata_cols = [col for col in df_history.columns 
                    if col.endswith('_encoded') and col in df_history.columns]
    
    if not metadata_cols:
        # Use basic numerical features if encoded features not available
        metadata_cols = ['age_at_study']
    
    print(f"Target columns: {target_cols}")
    print(f"Metadata columns: {metadata_cols}")
    
    # Split data (only current visits)
    current_df = df_history[df_history['visit_year'] == 0].copy()
    
    train_df, test_df = train_test_split(
        current_df, test_size=config.TEST_SIZE, random_state=42, 
        stratify=current_df['x_case']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=config.VAL_SIZE/(1-config.TEST_SIZE), random_state=42,
        stratify=train_df['x_case']
    )
    
    print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Store results for different history configurations
    all_results = {}
    
    # Test different history lengths
    for history_years in config.HISTORY_CONFIGS:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {history_years} YEARS OF HISTORY")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = HistoryAwareBreastCancerDataset(
            df_history, image_paths, target_cols, metadata_cols, 
            history_years=history_years, transform=train_transform
        )
        
        val_dataset = HistoryAwareBreastCancerDataset(
            df_history, image_paths, target_cols, metadata_cols, 
            history_years=history_years, transform=train_transform
        )
        
        test_dataset = HistoryAwareBreastCancerDataset(
            df_history, image_paths, target_cols, metadata_cols, 
            history_years=history_years, transform=train_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        # Initialize model
        model = HistoryAwareTransformer(
            num_metadata_features=len(metadata_cols) * (history_years + 1),
            history_years=history_years,
            num_classes=len(target_cols)
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        trained_model, history = train_history_aware_model(
            model, train_loader, val_loader, device, config.NUM_EPOCHS
        )
        
        # Evaluate model
        test_metrics, test_patient_ids = evaluate_history_model(
            trained_model, test_loader, device
        )
        
        # Store results
        all_results[f'{history_years}_years'] = {
            'metrics': test_metrics,
            'model': trained_model,
            'history': history
        }
        
        # Print results for this configuration
        print(f"\nResults for {history_years} years of history:")
        print("-" * 50)
        
        if test_metrics:
            for horizon, metrics in test_metrics.items():
                auc = metrics['auc']
                c_index = metrics['c_index']
                print(f"{horizon}:")
                print(f"  AUC: {auc:.3f} (95% CI: {metrics['auc_ci_lower']:.3f}-{metrics['auc_ci_upper']:.3f})")
                print(f"  C-Index: {c_index:.3f} (95% CI: {metrics['c_index_ci_lower']:.3f}-{metrics['c_index_ci_upper']:.3f})")
        else:
            print("  No valid metrics computed")
    
    # Generate comparative analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS: IS MORE THE MERRIER?")
    print(f"{'='*80}")
    
    # Create results tables
    create_history_comparison_tables(all_results)
    
    # Create visualizations
    create_history_comparison_plots(all_results)
    
    print(f"\n{'='*80}")
    print("MAMMOGRAPHY HISTORY STUDY COMPLETE!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def create_synthetic_history_data():
    """Create synthetic data for testing when real data is not available"""
    print("Creating synthetic history data for demonstration...")
    
    np.random.seed(42)
    n_patients = 100
    
    # Create base patient data
    data = []
    for patient_id in range(n_patients):
        base_age = np.random.randint(40, 80)
        cancer_status = np.random.choice([0, 1], p=[0.9, 0.1])
        
        # Current visit
        for visit_year in range(-4, 1):  # -4, -3, -2, -1, 0
            record = {
                'patient_id': patient_id,
                'anon_filename': f'patient_{patient_id}_year_{visit_year}.dcm',
                'visit_year': visit_year,
                'age_at_study': base_age - visit_year,  # Older in past visits
                'x_case': cancer_status if visit_year == 0 else 0,  # Cancer only in current
                'menopause_status': np.random.choice(['pre', 'post']),
                'hrt_status': np.random.choice(['never', 'current', 'former']),
                'family_history_breast': np.random.choice([0, 1]),
                'years_to_cancer': np.random.choice([1, 2, 3, 4, 5]) if cancer_status else 999
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    
    # Create target variables
    for years in [1, 2, 3, 4]:
        df[f'cancer_{years}year'] = (
            (df['x_case'] == 1) & (df['years_to_cancer'] <= years)
        ).astype(int)
    
    # Encode categorical variables
    le_dict = {}
    for col in ['menopause_status', 'hrt_status']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le
    
    print(f"Created synthetic dataset: {len(df)} records for {n_patients} patients")
    return df

def create_history_comparison_tables(all_results):
    """Create comparison tables for different history lengths"""
    
    print("\nHISTORY LENGTH COMPARISON TABLES")
    print("="*80)
    
    # AUC Comparison Table
    print("\nAUC COMPARISON (Area Under ROC Curve)")
    print("-" * 80)
    
    auc_data = []
    for config_name, results in all_results.items():
        history_years = config_name.replace('_years', '')
        row = [f"{history_years} years"]
        
        if 'metrics' in results and results['metrics']:
            for horizon in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']:
                if horizon in results['metrics']:
                    metrics = results['metrics'][horizon]
                    auc = metrics['auc']
                    ci_lower = metrics['auc_ci_lower']
                    ci_upper = metrics['auc_ci_upper']
                    row.append(f"{auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
                else:
                    row.append("N/A")
        else:
            row.extend(["N/A"] * 4)
        
        auc_data.append(row)
    
    auc_df = pd.DataFrame(auc_data, columns=['History Length', '1-Year', '2-Year', '3-Year', '4-Year'])
    print(auc_df.to_string(index=False))
    
    # C-Index Comparison Table
    print("\n\nC-INDEX COMPARISON (Concordance Index)")
    print("-" * 80)
    
    c_index_data = []
    for config_name, results in all_results.items():
        history_years = config_name.replace('_years', '')
        row = [f"{history_years} years"]
        
        if 'metrics' in results and results['metrics']:
            for horizon in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']:
                if horizon in results['metrics']:
                    metrics = results['metrics'][horizon]
                    c_index = metrics['c_index']
                    ci_lower = metrics['c_index_ci_lower']
                    ci_upper = metrics['c_index_ci_upper']
                    row.append(f"{c_index:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
                else:
                    row.append("N/A")
        else:
            row.extend(["N/A"] * 4)
        
        c_index_data.append(row)
    
    c_index_df = pd.DataFrame(c_index_data, columns=['History Length', '1-Year', '2-Year', '3-Year', '4-Year'])
    print(c_index_df.to_string(index=False))
    
    # Summary Statistics
    print("\n\nSUMMARY STATISTICS")
    print("-" * 80)
    
    summary_data = []
    for config_name, results in all_results.items():
        history_years = config_name.replace('_years', '')
        
        if 'metrics' in results and results['metrics']:
            aucs = [results['metrics'][h]['auc'] for h in results['metrics']]
            c_indices = [results['metrics'][h]['c_index'] for h in results['metrics']]
            
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            mean_c_index = np.mean(c_indices)
            std_c_index = np.std(c_indices)
            
            summary_data.append([
                f"{history_years} years",
                f"{mean_auc:.3f} ± {std_auc:.3f}",
                f"{mean_c_index:.3f} ± {std_c_index:.3f}"
            ])
        else:
            summary_data.append([f"{history_years} years", "N/A", "N/A"])
    
    summary_df = pd.DataFrame(summary_data, columns=['History Length', 'Mean AUC ± SD', 'Mean C-Index ± SD'])
    print(summary_df.to_string(index=False))
    
    # Save results
    try:
        auc_df.to_csv('history_study_auc_comparison.csv', index=False)
        c_index_df.to_csv('history_study_c_index_comparison.csv', index=False)
        summary_df.to_csv('history_study_summary.csv', index=False)
        print("\nComparison tables saved to CSV files!")
    except Exception as e:
        print(f"Error saving CSV files: {e}")

def create_history_comparison_plots(all_results):
    """Create visualization plots comparing different history lengths"""
    
    try:
        print("\nCreating comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        history_lengths = []
        mean_aucs = []
        mean_c_indices = []
        auc_stds = []
        c_index_stds = []
        
        for config_name, results in all_results.items():
            history_years = int(config_name.replace('_years', ''))
            history_lengths.append(history_years)
            
            if 'metrics' in results and results['metrics']:
                aucs = [results['metrics'][h]['auc'] for h in results['metrics']]
                c_indices = [results['metrics'][h]['c_index'] for h in results['metrics']]
                
                mean_aucs.append(np.mean(aucs))
                auc_stds.append(np.std(aucs))
                mean_c_indices.append(np.mean(c_indices))
                c_index_stds.append(np.std(c_indices))
            else:
                mean_aucs.append(0.5)
                auc_stds.append(0)
                mean_c_indices.append(0.5)
                c_index_stds.append(0)
        
        # Sort by history length
        sorted_data = sorted(zip(history_lengths, mean_aucs, auc_stds, mean_c_indices, c_index_stds))
        history_lengths, mean_aucs, auc_stds, mean_c_indices, c_index_stds = zip(*sorted_data)
        
        # Plot 1: Mean AUC vs History Length
        axes[0, 0].errorbar(history_lengths, mean_aucs, yerr=auc_stds, 
                           marker='o', capsize=5, capthick=2, linewidth=2)
        axes[0, 0].set_xlabel('Years of History')
        axes[0, 0].set_ylabel('Mean AUC')
        axes[0, 0].set_title('Mean AUC vs Mammography History Length')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0.5, 1.0])
        
        # Plot 2: Mean C-Index vs History Length  
        axes[0, 1].errorbar(history_lengths, mean_c_indices, yerr=c_index_stds,
                           marker='s', capsize=5, capthick=2, linewidth=2, color='red')
        axes[0, 1].set_xlabel('Years of History')
        axes[0, 1].set_ylabel('Mean C-Index')
        axes[0, 1].set_title('Mean C-Index vs Mammography History Length')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.5, 1.0])
        
        # Plot 3: AUC for different prediction horizons
        horizons = ['1-Year', '2-Year', '3-Year', '4-Year']
        horizon_keys = ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']
        
        for i, (horizon, horizon_key) in enumerate(zip(horizons, horizon_keys)):
            horizon_aucs = []
            for config_name, results in all_results.items():
                if 'metrics' in results and results['metrics'] and horizon_key in results['metrics']:
                    horizon_aucs.append(results['metrics'][horizon_key]['auc'])
                else:
                    horizon_aucs.append(0.5)
            
            axes[1, 0].plot(history_lengths, horizon_aucs, marker='o', label=horizon, linewidth=2)
        
        axes[1, 0].set_xlabel('Years of History')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('AUC by Prediction Horizon vs History Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0.5, 1.0])
        
        # Plot 4: Performance improvement over baseline (0 years history)
        if mean_aucs:
            baseline_auc = mean_aucs[0] if history_lengths[0] == 0 else mean_aucs[0]
            auc_improvements = [(auc - baseline_auc) * 100 for auc in mean_aucs]
            
            axes[1, 1].bar(history_lengths, auc_improvements, alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Years of History')
            axes[1, 1].set_ylabel('AUC Improvement (%)')
            axes[1, 1].set_title('Performance Improvement Over Baseline')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('mammography_history_study_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison plots saved as 'mammography_history_study_results.png'")
        
    except Exception as e:
        print(f"Error creating plots: {e}")

# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================

if __name__ == "__main__":
    run_history_study()
ibility
torch.manual_seed(42)
np.random.seed(42)

print("="*80)
print("MAMMOGRAPHY HISTORY STUDY: IS MORE THE MERRIER?")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ===========================================================================================
# CONFIGURATION
# ===========================================================================================

class Config:
    """Configuration parameters"""
    # Dataset paths (adjust these to your Kaggle dataset paths)
    METADATA_PATH = "/kaggle/input/breast-cancer-research-metadata/CSAW-CC_breast_cancer_screening_data.csv"
    IMAGES_PATH = "/kaggle/input/breast-cancer-research-dataset-batch-1-and-batch-2"
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 15  # Reduced for multiple experiments
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224
    
    # Model parameters
    NUM_CLASSES = 1  # Binary classification for each time horizon
    DROPOUT_RATE = 0.3
    
    # History study parameters
    MAX_HISTORY_YEARS = 4
    HISTORY_CONFIGS = [0, 1, 2, 3, 4]  # Number of historical years to include
    
    # Other parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    N_BOOTSTRAP = 1000

config = Config()

def concordance_index(targets, predictions):
    """
    Calculate C-index (concordance index) manually.
    For binary classification, C-index is equivalent to AUC.
    """
    try:
        return roc_auc_score(targets, predictions)
    except:
        return 0.5  # Random performance if calculation fails

# ===========================================================================================
# DATA PREPROCESSING WITH HISTORY SIMULATION
# ===========================================================================================

def simulate_patient_history(df, max_years=4):
    """
    Simulate mammography history for each patient.
    In real data, you would have actual historical images.
    For simulation, we create synthetic historical data based on current data.
    """
    print(f"\nSimulating {max_years} years of mammography history...")
    
    # Create patient groups (simulate patients having multiple visits)
    unique_patients = df.groupby(['age_at_study', 'menopause_status']).first().reset_index()
    n_patients = len(unique_patients)
    
    # Assign each original record to a patient
    np.random.seed(42)
    df['patient_id'] = np.random.randint(0, n_patients//3, len(df))  # Some patients have multiple visits
    
    # Create historical records for each patient
    history_data = []
    
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].iloc[0].copy()
        
        # Current visit (year 0)
        current_record = patient_data.copy()
        current_record['visit_year'] = 0
        current_record['years_to_cancer'] = np.random.choice([1, 2, 3, 4, 5]) if current_record['x_case'] == 1 else 999
        history_data.append(current_record)
        
        # Historical visits (years -1, -2, -3, -4)
        for year_back in range(1, max_years + 1):
            historical_record = patient_data.copy()
            historical_record['visit_year'] = -year_back
            historical_record['age_at_study'] = max(18, historical_record['age_at_study'] - year_back)
            
            # Simulate changes in historical data
            # Mammographic density might change over time
            if 'mammo_density' in historical_record.index:
                # Density tends to decrease with age
                density_change = np.random.normal(-0.1 * year_back, 0.05)
                historical_record['mammo_density'] = max(1, historical_record['mammo_density'] + density_change)
            
            # Cancer status - only current visit can be cancer positive for prediction
            historical_record['x_case'] = 0
            historical_record['years_to_cancer'] = historical_record['years_to_cancer'] + year_back
            
            history_data.append(historical_record)
    
    history_df = pd.DataFrame(history_data)
    print(f"Created history dataset: {len(history_df)} records for {len(history_df['patient_id'].unique())} patients")
    print(f"Visit years distribution:\n{history_df['visit_year'].value_counts().sort_index()}")
    
    return history_df

def load_and_preprocess_metadata_with_history(file_path, max_history_years=4):
    """Load and preprocess metadata with mammography history simulation"""
    print("Loading and preprocessing metadata with history...")
    
    # Load base dataset
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Original dataset shape: {df.shape}")
    
    # Clean data
    df_clean = df.dropna(subset=['anon_filename'])
    df_clean = df_clean[df_clean['anon_filename'].str.contains('.dcm', na=False)]
    print(f"After cleaning: {df_clean.shape}")
    
    # Select relevant columns
    feature_cols = [
        'anon_filename', 'age_at_study', 'x_case',
        'menopause_status', 'hrt_status', 'birads_density',
        'family_history_breast', 'family_history_ovarian',
        'previous_benign_biopsy', 'previous_cancer'
    ]
    
    available_cols = [col for col in feature_cols if col in df_clean.columns]
    df_subset = df_clean[available_cols].copy()
    print(f"Selected features: {available_cols}")
    
    # Handle missing values
    categorical_cols = ['menopause_status', 'hrt_status', 'family_history_breast', 
                       'family_history_ovarian', 'previous_benign_biopsy', 'previous_cancer']
    
    for col in categorical_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].fillna('Unknown')
    
    # Simulate patient history
    history_df = simulate_patient_history(df_subset, max_history_years)
    
    # Create target variables for different prediction horizons
    print("\nCreating target variables...")
    for years in [1, 2, 3, 4]:
        # Target: will patient develop cancer within X years?
        history_df[f'cancer_{years}year'] = (
            (history_df['x_case'] == 1) & 
            (history_df['years_to_cancer'] <= years)
        ).astype(int)
        
        print(f"  cancer_{years}year: {history_df[f'cancer_{years}year'].sum()} positive cases")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    le_dict = {}
    for col in categorical_cols:
        if col in history_df.columns:
            le = LabelEncoder()
            history_df[f'{col}_encoded'] = le.fit_transform(history_df[col].astype(str))
            le_dict[col] = le
            print(f"  {col}: encoded to {col}_encoded")
    
    return history_df, le_dict

# ===========================================================================================
# HISTORY-AWARE DATASET CLASS
# ===========================================================================================

class HistoryAwareBreastCancerDataset(Dataset):
    """Dataset class that handles different amounts of mammography history"""
    
    def __init__(self, df, image_paths, target_cols, metadata_cols, history_years=0, transform=None):
        """
        Args:
            df: DataFrame with patient data including history
            image_paths: Dictionary mapping filenames to paths
            target_cols: List of target column names
            metadata_cols: List of metadata column names
            history_years: Number of historical years to include (0-4)
            transform: Image transforms
        """
        self.df = df
        self.image_paths = image_paths
        self.target_cols = target_cols
        self.metadata_cols = metadata_cols
        self.history_years = history_years
        self.transform = transform
        
        # Group by patient and visit year for history access
        self.patient_data = {}
        for idx, row in df.iterrows():
            patient_id = row['patient_id']
            visit_year = row['visit_year']
            
            if patient_id not in self.patient_data:
                self.patient_data[patient_id] = {}
            self.patient_data[patient_id][visit_year] = row
        
        # Create samples list (only current visits for prediction)
        self.samples = []
        for idx, row in df.iterrows():
            if row['visit_year'] == 0:  # Only current visits
                self.samples.append(row)
        
        print(f"Dataset with {history_years} years history: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        current_sample = self.samples[idx]
        patient_id = current_sample['patient_id']
        
        # Collect images and metadata for current + historical visits
        images = []
        metadata_features = []
        
        # Current visit (year 0)
        current_image = self.load_image(current_sample['anon_filename'])
        if current_image is not None:
            images.append(current_image)
        
        current_metadata = [current_sample[col] for col in self.metadata_cols 
                          if not pd.isna(current_sample[col])]
        metadata_features.extend(current_metadata)
        
        # Historical visits (years -1, -2, ..., -history_years)
        for year_back in range(1, self.history_years + 1):
            visit_year = -year_back
            if (patient_id in self.patient_data and 
                visit_year in self.patient_data[patient_id]):
                
                historical_sample = self.patient_data[patient_id][visit_year]
                
                # Load historical image
                historical_image = self.load_image(historical_sample['anon_filename'])
                if historical_image is not None:
                    images.append(historical_image)
                
                # Add historical metadata
                historical_metadata = [historical_sample[col] for col in self.metadata_cols 
                                     if not pd.isna(historical_sample[col])]
                metadata_features.extend(historical_metadata)
        
        # Pad or truncate to fixed size
        max_images = self.history_years + 1
        while len(images) < max_images:
            # Duplicate last image if not enough history
            if images:
                images.append(images[-1])
            else:
                # Create dummy image if no images available
                dummy_image = torch.zeros(3, 224, 224)
                images.append(dummy_image)
        
        images = images[:max_images]  # Truncate if too many
        
        # Stack images (shape: [history_length, 3, 224, 224])
        image_tensor = torch.stack(images)
        
        # Metadata features
        if metadata_features:
            metadata_tensor = torch.tensor(metadata_features, dtype=torch.float32)
        else:
            metadata_tensor = torch.zeros(len(self.metadata_cols) * max_images, dtype=torch.float32)
        
        # Targets
        targets = torch.tensor([current_sample[col] for col in self.target_cols], dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'metadata': metadata_tensor,
            'targets': targets,
            'patient_id': patient_id
        }
    
    def load_image(self, filename):
        """Load and preprocess a single DICOM image"""
        if filename not in self.image_paths or pd.isna(filename):
            return None
        
        try:
            # Load DICOM
            dicom_path = self.image_paths[filename]
            dicom_data = pydicom.dcmread(dicom_path)
            image_array = dicom_data.pixel_array
            
            # Normalize to 0-255
            image_array = ((image_array - image_array.min()) / 
                          (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            
            # Convert to 3-channel
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            
            # Convert to PIL and resize
            pil_image = Image.fromarray(image_array)
            pil_image = pil_image.resize((224, 224))
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(pil_image)
            else:
                image_tensor = transforms.ToTensor()(pil_image)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return torch.zeros(3, 224, 224)

# ===========================================================================================
# HISTORY-AWARE MODEL ARCHITECTURE
# ===========================================================================================

class HistoryAwareTransformer(nn.Module):
    """Transformer model that processes multiple mammograms with temporal awareness"""
    
    def __init__(self, num_metadata_features, history_years=0, num_classes=4):
        super(HistoryAwareTransformer, self).__init__()
        
        self.history_years = history_years
        self.num_images = history_years + 1  # Current + historical
        
        # Vision Transformer for each image
        self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.image_feature_dim = self.image_encoder.num_features
        
        # Remove the classification head
        self.image_encoder.head = nn.Identity()
        
        # Temporal encoding for different visits
        self.temporal_embedding = nn.Embedding(self.num_images, self.image_feature_dim)
        
        # Transformer encoder for temporal fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.image_feature_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Metadata processing
        self.metadata_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        fusion_dim = self.image_feature_dim + 64
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Multi-task prediction heads
        self.prediction_heads = nn.ModuleDict({
            f'cancer_{i}year': nn.Linear(256, 1) for i in [1, 2, 3, 4]
        })
    
    def forward(self, images, metadata):
        batch_size = images.shape[0]
        
        # Process each image through ViT
        # images shape: [batch_size, num_images, 3, 224, 224]
        image_features = []
        for i in range(self.num_images):
            if i < images.shape[1]:
                img_feat = self.image_encoder(images[:, i])  # [batch_size, feature_dim]
            else:
                # Pad with zeros if not enough images
                img_feat = torch.zeros(batch_size, self.image_feature_dim, 
                                     device=images.device, dtype=images.dtype)
            image_features.append(img_feat)
        
        # Stack and add temporal embeddings
        image_features = torch.stack(image_features, dim=1)  # [batch_size, num_images, feature_dim]
        
        # Add temporal position embeddings
        temporal_positions = torch.arange(self.num_images, device=images.device)
        temporal_embeds = self.temporal_embedding(temporal_positions)  # [num_images, feature_dim]
        temporal_embeds = temporal_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_images, feature_dim]
        
        image_features = image_features + temporal_embeds
        
        # Apply temporal transformer
        temporal_features = self.temporal_transformer(image_features)  # [batch_size, num_images, feature_dim]
        
        # Global pooling over time dimension
        pooled_features = temporal_features.mean(dim=1)  # [batch_size, feature_dim]
        
        # Process metadata
        metadata_features = self.metadata_mlp(metadata)  # [batch_size, 64]
        
        # Fusion
        fused_features = torch.cat([pooled_features, metadata_features], dim=1)
        fused_features = self.fusion_layer(fused_features)  # [batch_size, 256]
        
        # Multi-task predictions
        predictions = {}
        for task_name, head in self.prediction_heads.items():
            predictions[task_name] = torch.sigmoid(head(fused_features))
        
        return predictions

# ===========================================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ===========================================================================================

def train_history_aware_model(model, train_loader, val_loader, device, num_epochs=15):
    """Train the history-aware model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\nTraining history-aware model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            try:
                images = batch['image'].to(device, non_blocking=True)
                metadata = batch['metadata'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(images, metadata)
                
                # Calculate multi-task loss
                total_loss = 0
                for i, task_name in enumerate(predictions.keys()):
                    task_pred = predictions[task_name].squeeze()
                    task_target = targets[:, i]
                    task_loss = criterion(task_pred, task_target)
                    total_loss += task_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
                
                if batch_idx == 0:  # First batch of epoch
                    print(f"  Epoch {epoch+1}/{num_epochs}, Batch 1 - Loss: {total_loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                try:
                    images = batch['image'].to(device, non_blocking=True)
                    metadata = batch['metadata'].to(device, non_blocking=True)
                    targets = batch['targets'].to(device, non_blocking=True)
                    
                    predictions = model(images, metadata)
                    
                    total_loss = 0
                    for i, task_name in enumerate(predictions.keys()):
                        task_pred = predictions[task_name].squeeze()
                        task_target = targets[:, i]
                        task_loss = criterion(task_pred, task_target)
                        total_loss += task_loss
                    
                    val_loss += total_loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def evaluate_history_model(model, test_loader, device):
    """Evaluate the history-aware model"""
    
    model.eval()
    all_predictions = {f'cancer_{i}year': [] for i in [1, 2, 3, 4]}
    all_targets = {f'cancer_{i}year': [] for i in [1, 2, 3, 4]}
    patient_ids = []
    
    print("Evaluating history-aware model...")
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
                
            try:
                images = batch['image'].to(device, non_blocking=True)
                metadata = batch['metadata'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                batch_patient_ids = batch['patient_id']
                
                predictions = model(images, metadata)
                
                # Store predictions and targets
                for i, task_name in enumerate(predictions.keys()):
                    pred = predictions[task_name].squeeze().cpu().numpy()
                    target = targets[:, i].cpu().numpy()
                    
                    if pred.ndim == 0:
                        pred = pred.reshape(1)
                        target = target.reshape(1)
                    
                    all_predictions[task_name].extend(pred)
                    all_targets[task_name].extend(target)
                
                patient_ids.extend(batch_patient_ids)
                
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    # Calculate metrics
    metrics = {}
    for task_name in all_predictions.keys():
        if len(all_predictions[task_name]) > 0 and len(all_targets[task_name]) > 0:
            preds = np.array(all_predictions[task_name])
            targets = np.array(all_targets[task_name])
            
            # Skip if no positive cases
            if len(np.unique(targets)) > 1:
                auc = roc_auc_score(targets, preds)
                
                # Calculate C-index (same as AUC for binary classification)
                c_index = concordance_index(targets, preds)
                
                # Bootstrap confidence intervals
                aucs = []
                c_indices = []
                for _ in range(100):  # Reduced for speed
                    indices = np.random.choice(len(targets), len(targets), replace=True)
                    if len(np.unique(targets[indices])) > 1:
                        bootstrap_auc = roc_auc_score(targets[indices], preds[indices])
                        bootstrap_c_index = concordance_index(targets[indices], preds[indices])
                        aucs.append(bootstrap_auc)
                        c_indices.append(bootstrap_c_index)
                
                auc_ci = np.percentile(aucs, [2.5, 97.5]) if aucs else [auc, auc]
                c_index_ci = np.percentile(c_indices, [2.5, 97.5]) if c_indices else [c_index, c_index]
                
                metrics[task_name] = {
                    'auc': auc,
                    'auc_ci_lower': auc_ci[0],
                    'auc_ci_upper': auc_ci[1],
                    'c_index': c_index,
                    'c_index_ci_lower': c_index_ci[0],
                    'c_index_ci_upper': c_index_ci[1],
                    'predictions': preds,
                    'targets': targets
                }
    
    return metrics, patient_ids

# ===========================================================================================
# MAIN HISTORY STUDY PIPELINE
# ===========================================================================================

def run_history_study():
    """Run the complete mammography history study"""
    
    print("="*80)
    print("STARTING MAMMOGRAPHY HISTORY STUDY")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Load data with history simulation
    try:
        df_history, le_dict = load_and_preprocess_metadata_with_history(
            config.METADATA_PATH, 
            max_history_years=config.MAX_HISTORY_YEARS
        )
    except FileNotFoundError:
        print("Dataset not found. Creating synthetic dataset for demonstration...")
        # Create synthetic data for testing
        df_history = create_synthetic_history_data()
        le_dict = {}
    
    # Get image paths (simulated)
    print("\nSetting up image paths...")
    image_paths = {}
    unique_filenames = df_history['anon_filename'].unique()
    for filename in unique_filenames:
        # In real implementation, map to actual DICOM files
        image_paths[filename] = f"synthetic_path/{filename}"
    
    # Prepare features
    target_cols = ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']
    metadata_cols = [col for col in df_history.columns 
                    if col.endswith('_encoded') and col in df_history.columns]
    
    if not metadata_cols:
        # Use basic numerical features if encoded features not available
        metadata_cols = ['age_at_study']
    
    print(f"Target columns: {target_cols}")
    print(f"Metadata columns: {metadata_cols}")
    
    # Split data (only current visits)
    current_df = df_history[df_history['visit_year'] == 0].copy()
    
    train_df, test_df = train_test_split(
        current_df, test_size=config.TEST_SIZE, random_state=42, 
        stratify=current_df['x_case']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=config.VAL_SIZE/(1-config.TEST_SIZE), random_state=42,
        stratify=train_df['x_case']
    )
    
    print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Store results for different history configurations
    all_results = {}
    
    # Test different history lengths
    for history_years in config.HISTORY_CONFIGS:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {history_years} YEARS OF HISTORY")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = HistoryAwareBreastCancerDataset(
            df_history, image_paths, target_cols, metadata_cols, 
            history_years=history_years, transform=train_transform
        )
        
        val_dataset = HistoryAwareBreastCancerDataset(
            df_history, image_paths, target_cols, metadata_cols, 
            history_years=history_years, transform=train_transform
        )
        
        test_dataset = HistoryAwareBreastCancerDataset(
            df_history, image_paths, target_cols, metadata_cols, 
            history_years=history_years, transform=train_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        # Initialize model
        model = HistoryAwareTransformer(
            num_metadata_features=len(metadata_cols) * (history_years + 1),
            history_years=history_years,
            num_classes=len(target_cols)
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        trained_model, history = train_history_aware_model(
            model, train_loader, val_loader, device, config.NUM_EPOCHS
        )
        
        # Evaluate model
        test_metrics, test_patient_ids = evaluate_history_model(
            trained_model, test_loader, device
        )
        
        # Store results
        all_results[f'{history_years}_years'] = {
            'metrics': test_metrics,
            'model': trained_model,
            'history': history
        }
        
        # Print results for this configuration
        print(f"\nResults for {history_years} years of history:")
        print("-" * 50)
        
        if test_metrics:
            for horizon, metrics in test_metrics.items():
                auc = metrics['auc']
                c_index = metrics['c_index']
                print(f"{horizon}:")
                print(f"  AUC: {auc:.3f} (95% CI: {metrics['auc_ci_lower']:.3f}-{metrics['auc_ci_upper']:.3f})")
                print(f"  C-Index: {c_index:.3f} (95% CI: {metrics['c_index_ci_lower']:.3f}-{metrics['c_index_ci_upper']:.3f})")
        else:
            print("  No valid metrics computed")
    
    # Generate comparative analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS: IS MORE THE MERRIER?")
    print(f"{'='*80}")
    
    # Create results tables
    create_history_comparison_tables(all_results)
    
    # Create visualizations
    create_history_comparison_plots(all_results)
    
    print(f"\n{'='*80}")
    print("MAMMOGRAPHY HISTORY STUDY COMPLETE!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def create_synthetic_history_data():
    """Create synthetic data for testing when real data is not available"""
    print("Creating synthetic history data for demonstration...")
    
    np.random.seed(42)
    n_patients = 100
    
    # Create base patient data
    data = []
    for patient_id in range(n_patients):
        base_age = np.random.randint(40, 80)
        cancer_status = np.random.choice([0, 1], p=[0.9, 0.1])
        
        # Current visit
        for visit_year in range(-4, 1):  # -4, -3, -2, -1, 0
            record = {
                'patient_id': patient_id,
                'anon_filename': f'patient_{patient_id}_year_{visit_year}.dcm',
                'visit_year': visit_year,
                'age_at_study': base_age - visit_year,  # Older in past visits
                'x_case': cancer_status if visit_year == 0 else 0,  # Cancer only in current
                'menopause_status': np.random.choice(['pre', 'post']),
                'hrt_status': np.random.choice(['never', 'current', 'former']),
                'family_history_breast': np.random.choice([0, 1]),
                'years_to_cancer': np.random.choice([1, 2, 3, 4, 5]) if cancer_status else 999
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    
    # Create target variables
    for years in [1, 2, 3, 4]:
        df[f'cancer_{years}year'] = (
            (df['x_case'] == 1) & (df['years_to_cancer'] <= years)
        ).astype(int)
    
    # Encode categorical variables
    le_dict = {}
    for col in ['menopause_status', 'hrt_status']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le
    
    print(f"Created synthetic dataset: {len(df)} records for {n_patients} patients")
    return df

def create_history_comparison_tables(all_results):
    """Create comparison tables for different history lengths"""
    
    print("\nHISTORY LENGTH COMPARISON TABLES")
    print("="*80)
    
    # AUC Comparison Table
    print("\nAUC COMPARISON (Area Under ROC Curve)")
    print("-" * 80)
    
    auc_data = []
    for config_name, results in all_results.items():
        history_years = config_name.replace('_years', '')
        row = [f"{history_years} years"]
        
        if 'metrics' in results and results['metrics']:
            for horizon in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']:
                if horizon in results['metrics']:
                    metrics = results['metrics'][horizon]
                    auc = metrics['auc']
                    ci_lower = metrics['auc_ci_lower']
                    ci_upper = metrics['auc_ci_upper']
                    row.append(f"{auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
                else:
                    row.append("N/A")
        else:
            row.extend(["N/A"] * 4)
        
        auc_data.append(row)
    
    auc_df = pd.DataFrame(auc_data, columns=['History Length', '1-Year', '2-Year', '3-Year', '4-Year'])
    print(auc_df.to_string(index=False))
    
    # C-Index Comparison Table
    print("\n\nC-INDEX COMPARISON (Concordance Index)")
    print("-" * 80)
    
    c_index_data = []
    for config_name, results in all_results.items():
        history_years = config_name.replace('_years', '')
        row = [f"{history_years} years"]
        
        if 'metrics' in results and results['metrics']:
            for horizon in ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']:
                if horizon in results['metrics']:
                    metrics = results['metrics'][horizon]
                    c_index = metrics['c_index']
                    ci_lower = metrics['c_index_ci_lower']
                    ci_upper = metrics['c_index_ci_upper']
                    row.append(f"{c_index:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
                else:
                    row.append("N/A")
        else:
            row.extend(["N/A"] * 4)
        
        c_index_data.append(row)
    
    c_index_df = pd.DataFrame(c_index_data, columns=['History Length', '1-Year', '2-Year', '3-Year', '4-Year'])
    print(c_index_df.to_string(index=False))
    
    # Summary Statistics
    print("\n\nSUMMARY STATISTICS")
    print("-" * 80)
    
    summary_data = []
    for config_name, results in all_results.items():
        history_years = config_name.replace('_years', '')
        
        if 'metrics' in results and results['metrics']:
            aucs = [results['metrics'][h]['auc'] for h in results['metrics']]
            c_indices = [results['metrics'][h]['c_index'] for h in results['metrics']]
            
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            mean_c_index = np.mean(c_indices)
            std_c_index = np.std(c_indices)
            
            summary_data.append([
                f"{history_years} years",
                f"{mean_auc:.3f} ± {std_auc:.3f}",
                f"{mean_c_index:.3f} ± {std_c_index:.3f}"
            ])
        else:
            summary_data.append([f"{history_years} years", "N/A", "N/A"])
    
    summary_df = pd.DataFrame(summary_data, columns=['History Length', 'Mean AUC ± SD', 'Mean C-Index ± SD'])
    print(summary_df.to_string(index=False))
    
    # Save results
    try:
        auc_df.to_csv('history_study_auc_comparison.csv', index=False)
        c_index_df.to_csv('history_study_c_index_comparison.csv', index=False)
        summary_df.to_csv('history_study_summary.csv', index=False)
        print("\nComparison tables saved to CSV files!")
    except Exception as e:
        print(f"Error saving CSV files: {e}")

def create_history_comparison_plots(all_results):
    """Create visualization plots comparing different history lengths"""
    
    try:
        print("\nCreating comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        history_lengths = []
        mean_aucs = []
        mean_c_indices = []
        auc_stds = []
        c_index_stds = []
        
        for config_name, results in all_results.items():
            history_years = int(config_name.replace('_years', ''))
            history_lengths.append(history_years)
            
            if 'metrics' in results and results['metrics']:
                aucs = [results['metrics'][h]['auc'] for h in results['metrics']]
                c_indices = [results['metrics'][h]['c_index'] for h in results['metrics']]
                
                mean_aucs.append(np.mean(aucs))
                auc_stds.append(np.std(aucs))
                mean_c_indices.append(np.mean(c_indices))
                c_index_stds.append(np.std(c_indices))
            else:
                mean_aucs.append(0.5)
                auc_stds.append(0)
                mean_c_indices.append(0.5)
                c_index_stds.append(0)
        
        # Sort by history length
        sorted_data = sorted(zip(history_lengths, mean_aucs, auc_stds, mean_c_indices, c_index_stds))
        history_lengths, mean_aucs, auc_stds, mean_c_indices, c_index_stds = zip(*sorted_data)
        
        # Plot 1: Mean AUC vs History Length
        axes[0, 0].errorbar(history_lengths, mean_aucs, yerr=auc_stds, 
                           marker='o', capsize=5, capthick=2, linewidth=2)
        axes[0, 0].set_xlabel('Years of History')
        axes[0, 0].set_ylabel('Mean AUC')
        axes[0, 0].set_title('Mean AUC vs Mammography History Length')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0.5, 1.0])
        
        # Plot 2: Mean C-Index vs History Length  
        axes[0, 1].errorbar(history_lengths, mean_c_indices, yerr=c_index_stds,
                           marker='s', capsize=5, capthick=2, linewidth=2, color='red')
        axes[0, 1].set_xlabel('Years of History')
        axes[0, 1].set_ylabel('Mean C-Index')
        axes[0, 1].set_title('Mean C-Index vs Mammography History Length')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.5, 1.0])
        
        # Plot 3: AUC for different prediction horizons
        horizons = ['1-Year', '2-Year', '3-Year', '4-Year']
        horizon_keys = ['cancer_1year', 'cancer_2year', 'cancer_3year', 'cancer_4year']
        
        for i, (horizon, horizon_key) in enumerate(zip(horizons, horizon_keys)):
            horizon_aucs = []
            for config_name, results in all_results.items():
                if 'metrics' in results and results['metrics'] and horizon_key in results['metrics']:
                    horizon_aucs.append(results['metrics'][horizon_key]['auc'])
                else:
                    horizon_aucs.append(0.5)
            
            axes[1, 0].plot(history_lengths, horizon_aucs, marker='o', label=horizon, linewidth=2)
        
        axes[1, 0].set_xlabel('Years of History')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('AUC by Prediction Horizon vs History Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0.5, 1.0])
        
        # Plot 4: Performance improvement over baseline (0 years history)
        if mean_aucs:
            baseline_auc = mean_aucs[0] if history_lengths[0] == 0 else mean_aucs[0]
            auc_improvements = [(auc - baseline_auc) * 100 for auc in mean_aucs]
            
            axes[1, 1].bar(history_lengths, auc_improvements, alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Years of History')
            axes[1, 1].set_ylabel('AUC Improvement (%)')
            axes[1, 1].set_title('Performance Improvement Over Baseline')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('mammography_history_study_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison plots saved as 'mammography_history_study_results.png'")
        
    except Exception as e:
        print(f"Error creating plots: {e}")

# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================

if __name__ == "__main__":
    run_history_study()
