import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from PIL import Image
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from lifelines.utils import concordance_index

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class Config:
    """Configuration class for the breast cancer prediction model."""
    
    # Dataset paths (UPDATE THESE FOR YOUR KAGGLE ENVIRONMENT)
    METADATA_PATH = "/kaggle/input/breast-cancer-research-metadata/CSAW-CC_breast_cancer_screening_data.csv"
    IMAGES_PATH = "/kaggle/input/breast-cancer-research-dataset-batch-1-and-batch-2"
    
    # Model parameters
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    EMBED_DIM = 768
    NUM_HEADS = 12
    NUM_LAYERS = 12
    MLP_DIM = 3072
    DROPOUT = 0.1
    
    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30
    WEIGHT_DECAY = 0.01
    
    # History settings
    MAX_HISTORY_YEARS = 4
    PREDICTION_YEARS = 5
    
    # Random seed
    SEED = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DICOMProcessor:
    """Handles DICOM image loading and preprocessing."""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_dicom(self, file_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess a DICOM file."""
        try:
            # Load DICOM file
            dicom = pydicom.dcmread(file_path)
            
            # Get pixel array
            if hasattr(dicom, 'pixel_array'):
                pixel_array = dicom.pixel_array
            else:
                logger.warning(f"No pixel array in {file_path}")
                return None
            
            # Normalize to 0-255 range
            pixel_array = pixel_array.astype(np.float32)
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
            pixel_array = (pixel_array * 255).astype(np.uint8)
            
            # Convert to PIL Image and apply transforms
            if len(pixel_array.shape) == 2:
                # Convert grayscale to RGB
                image = Image.fromarray(pixel_array).convert('RGB')
            else:
                image = Image.fromarray(pixel_array)
            
            # Apply transforms
            tensor = self.transform(image)
            return tensor
            
        except Exception as e:
            logger.error(f"Error loading DICOM {file_path}: {e}")
            return None

class DataProcessor:
    """Handles metadata processing and longitudinal sequence creation."""
    
    def __init__(self, metadata_path: str, images_path: str):
        self.metadata_path = metadata_path
        self.images_path = images_path
        self.dicom_processor = DICOMProcessor()
        
    def load_metadata(self) -> pd.DataFrame:
        """Load and clean metadata."""
        df = pd.read_csv(self.metadata_path)
        
        # Clean data
        df = df.dropna(subset=['anon_patientid', 'exam_year', 'anon_filename'])
        
        # Convert data types
        df['anon_patientid'] = df['anon_patientid'].astype(str)
        df['exam_year'] = df['exam_year'].astype(int)
        df['x_case'] = df['x_case'].fillna(0).astype(int)
        
        # Fill missing values for clinical features
        numeric_cols = ['libra_breastarea', 'libra_densearea', 'libra_percentdensity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        logger.info(f"Loaded metadata: {len(df)} records, {df['anon_patientid'].nunique()} patients")
        return df
    
    def create_visit_sequences(self, df: pd.DataFrame, history_years: int = 0) -> Dict:
        """Create longitudinal visit sequences for each patient."""
        
        # Group by patient and create visit sequences
        patient_sequences = {}
        
        for patient_id, patient_data in df.groupby('anon_patientid'):
            # Sort by exam year
            patient_data = patient_data.sort_values('exam_year')
            
            # Get unique exam years
            exam_years = sorted(patient_data['exam_year'].unique())
            
            # Create sequences for each possible present time point
            for i, present_year in enumerate(exam_years):
                # Skip if we don't have enough history
                if i < history_years:
                    continue
                
                # Get history years
                history_start_idx = max(0, i - history_years)
                history_years_list = exam_years[history_start_idx:i+1]
                
                # Get data for these years
                visit_data = []
                for year in history_years_list:
                    year_data = patient_data[patient_data['exam_year'] == year]
                    
                    # Group by views (should have 4 views per visit)
                    visit_images = []
                    visit_metadata = {}
                    
                    for _, row in year_data.iterrows():
                        # Store image path
                        visit_images.append(row['anon_filename'])
                        
                        # Store metadata (use latest for the year)
                        visit_metadata = {
                            'age': row.get('x_age', 0),
                            'breast_area': row.get('libra_breastarea', 0),
                            'dense_area': row.get('libra_densearea', 0),
                            'percent_density': row.get('libra_percentdensity', 0),
                            'cancer_outcome': row.get('x_case', 0)
                        }
                    
                    visit_data.append({
                        'year': year,
                        'images': visit_images,
                        'metadata': visit_metadata
                    })
                
                # Create outcome labels for future years
                future_outcomes = self._create_future_outcomes(
                    patient_data, present_year, Config.PREDICTION_YEARS
                )
                
                sequence_id = f"{patient_id}_{present_year}"
                patient_sequences[sequence_id] = {
                    'patient_id': patient_id,
                    'present_year': present_year,
                    'visits': visit_data,
                    'outcomes': future_outcomes,
                    'history_length': len(visit_data)
                }
        
        logger.info(f"Created {len(patient_sequences)} sequences with {history_years} years history")
        return patient_sequences
    
    def _create_future_outcomes(self, patient_data: pd.DataFrame, present_year: int, 
                               prediction_years: int) -> Dict[str, int]:
        """Create future cancer outcome labels."""
        outcomes = {}
        
        for year_offset in range(1, prediction_years + 1):
            target_year = present_year + year_offset
            
            # Check if cancer occurred in this target year or before
            future_data = patient_data[patient_data['exam_year'] <= target_year]
            has_cancer = (future_data['x_case'] == 1).any()
            
            outcomes[f'cancer_{year_offset}year'] = int(has_cancer)
        
        return outcomes
    
    def find_image_paths(self) -> Dict[str, str]:
        """Find all image file paths."""
        image_paths = {}
        
        # Search in both batch directories
        batch_dirs = ['Batch_1/Batch_1', 'Batch_2/Batch_2']
        
        for batch_dir in batch_dirs:
            batch_path = os.path.join(self.images_path, batch_dir)
            if os.path.exists(batch_path):
                for root, dirs, files in os.walk(batch_path):
                    for file in files:
                        if file.endswith('.dcm'):
                            image_paths[file] = os.path.join(root, file)
        
        logger.info(f"Found {len(image_paths)} DICOM images")
        return image_paths

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class VisionTransformer(nn.Module):
    """Vision Transformer for mammography image processing."""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12,
                 mlp_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Return class token
        return x[:, 0]

class ImageAggregator(nn.Module):
    """Aggregates multiple mammography views into single visit embedding."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # View type embedding (CC, MLO for Left/Right)
        self.view_embed = nn.Embedding(4, embed_dim)  # 4 views per mammogram
        
        # Transformer for aggregating views
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, view_embeddings):
        """
        Args:
            view_embeddings: (batch_size, num_views, embed_dim)
        """
        B, num_views, embed_dim = view_embeddings.shape
        
        # Add view type embeddings
        view_ids = torch.arange(num_views, device=view_embeddings.device)
        view_ids = view_ids.unsqueeze(0).expand(B, -1)
        view_embed = self.view_embed(view_ids)
        
        # Combine embeddings
        x = view_embeddings + view_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class VisitAggregator(nn.Module):
    """Aggregates multiple visits with temporal positional encoding."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Temporal positional encoding
        self.temporal_pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer for aggregating visits
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, visit_embeddings, visit_mask=None):
        """
        Args:
            visit_embeddings: (batch_size, num_visits, embed_dim)
            visit_mask: (batch_size, num_visits) - True for valid visits
        """
        B, num_visits, embed_dim = visit_embeddings.shape
        
        # Apply temporal positional encoding
        x = visit_embeddings.transpose(0, 1)  # (num_visits, batch_size, embed_dim)
        x = self.temporal_pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, num_visits, embed_dim)
        
        # Create attention mask if provided
        attn_mask = None
        if visit_mask is not None:
            attn_mask = ~visit_mask  # Transformer uses True for masked positions
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Aggregate (use last visit or mean)
        if visit_mask is not None:
            # Use masked mean
            mask_expanded = visit_mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded
            x = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class SurvivalModule(nn.Module):
    """Survival analysis module with cumulative hazard calculation."""
    
    def __init__(self, input_dim: int = 768, prediction_years: int = 5):
        super().__init__()
        
        self.prediction_years = prediction_years
        
        # Baseline risk
        self.baseline_risk = nn.Parameter(torch.zeros(1))
        
        # Year-specific hazard predictors
        self.hazard_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            ) for _ in range(prediction_years)
        ])
        
        # Additional clinical features integration
        self.clinical_proj = nn.Linear(4, 64)  # 4 clinical features
        self.fusion = nn.Linear(input_dim + 64, input_dim)
    
    def forward(self, visit_embedding, clinical_features):
        """
        Args:
            visit_embedding: (batch_size, embed_dim)
            clinical_features: (batch_size, 4) - [age, breast_area, dense_area, percent_density]
        """
        B = visit_embedding.shape[0]
        
        # Integrate clinical features
        clinical_embed = self.clinical_proj(clinical_features)
        combined = torch.cat([visit_embedding, clinical_embed], dim=-1)
        fused_embedding = self.fusion(combined)
        
        # Calculate year-specific hazards
        hazards = []
        for year_predictor in self.hazard_predictors:
            hazard = year_predictor(fused_embedding)
            hazards.append(hazard)
        
        # Calculate cumulative risks using the exact formula specified
        risks = {}
        cumulative_hazard = self.baseline_risk.expand(B, 1)
        
        for year in range(1, self.prediction_years + 1):
            cumulative_hazard = cumulative_hazard + hazards[year - 1]
            risk = torch.sigmoid(cumulative_hazard)
            risks[f'risk_{year}year'] = risk.squeeze(-1)
        
        return risks, hazards

class BreastCancerVisionTransformer(nn.Module):
    """Complete Vision Transformer model for breast cancer risk prediction."""
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        
        # Vision Transformer backbone
        self.vision_transformer = VisionTransformer(
            image_size=config.IMAGE_SIZE,
            patch_size=config.PATCH_SIZE,
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            mlp_dim=config.MLP_DIM,
            dropout=config.DROPOUT
        )
        
        # Image aggregator (4 views -> 1 visit embedding)
        self.image_aggregator = ImageAggregator(
            embed_dim=config.EMBED_DIM,
            num_heads=8,
            num_layers=4
        )
        
        # Visit aggregator (multiple visits -> patient embedding)
        self.visit_aggregator = VisitAggregator(
            embed_dim=config.EMBED_DIM,
            num_heads=8,
            num_layers=4
        )
        
        # Survival module
        self.survival_module = SurvivalModule(
            input_dim=config.EMBED_DIM,
            prediction_years=config.PREDICTION_YEARS
        )
    
    def forward(self, images, clinical_features, visit_mask=None):
        """
        Args:
            images: (batch_size, num_visits, num_views, 3, H, W)
            clinical_features: (batch_size, 4)
            visit_mask: (batch_size, num_visits)
        """
        B, num_visits, num_views, C, H, W = images.shape
        
        # Reshape for vision transformer processing
        images = images.view(B * num_visits * num_views, C, H, W)
        
        # Extract image features
        image_features = self.vision_transformer(images)  # (B*num_visits*num_views, embed_dim)
        
        # Reshape back
        image_features = image_features.view(B, num_visits, num_views, -1)
        
        # Aggregate views for each visit
        visit_embeddings = []
        for visit_idx in range(num_visits):
            visit_views = image_features[:, visit_idx]  # (B, num_views, embed_dim)
            visit_embedding = self.image_aggregator(visit_views)
            visit_embeddings.append(visit_embedding)
        
        visit_embeddings = torch.stack(visit_embeddings, dim=1)  # (B, num_visits, embed_dim)
        
        # Aggregate visits
        patient_embedding = self.visit_aggregator(visit_embeddings, visit_mask)
        
        # Generate survival predictions
        risks, hazards = self.survival_module(patient_embedding, clinical_features)
        
        return risks, hazards

class MammographyDataset(Dataset):
    """Dataset for mammography sequences."""
    
    def __init__(self, sequences: Dict, image_paths: Dict[str, str], 
                 history_years: int = 0, max_visits: int = 5):
        self.sequences = sequences
        self.image_paths = image_paths
        self.history_years = history_years
        self.max_visits = max_visits
        self.dicom_processor = DICOMProcessor()
        
        # Filter sequences that have required history
        self.valid_sequences = []
        for seq_id, seq_data in sequences.items():
            if len(seq_data['visits']) >= (history_years + 1):
                self.valid_sequences.append(seq_id)
        
        logger.info(f"Dataset created: {len(self.valid_sequences)} valid sequences")
    
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        seq_id = self.valid_sequences[idx]
        sequence = self.sequences[seq_id]
        
        # Get visits (limited by max_visits)
        visits = sequence['visits'][-self.max_visits:]
        
        # Load images for each visit
        visit_images = []
        visit_masks = []
        
        for visit in visits:
            # Load up to 4 images per visit
            view_tensors = []
            
            for img_filename in visit['images'][:4]:  # Max 4 views
                if img_filename in self.image_paths:
                    img_path = self.image_paths[img_filename]
                    img_tensor = self.dicom_processor.load_dicom(img_path)
                    
                    if img_tensor is not None:
                        view_tensors.append(img_tensor)
            
            # Pad to 4 views if necessary
            while len(view_tensors) < 4:
                if view_tensors:
                    view_tensors.append(view_tensors[-1].clone())  # Duplicate last view
                else:
                    # Create zero tensor if no views loaded
                    view_tensors.append(torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
            # Stack views
            visit_tensor = torch.stack(view_tensors[:4])  # (4, 3, H, W)
            visit_images.append(visit_tensor)
            visit_masks.append(True)
        
        # Pad visits to max_visits
        while len(visit_images) < self.max_visits:
            # Add dummy visit
            dummy_visit = torch.zeros(4, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            visit_images.append(dummy_visit)
            visit_masks.append(False)
        
        # Stack visits
        images = torch.stack(visit_images)  # (max_visits, 4, 3, H, W)
        visit_mask = torch.tensor(visit_masks, dtype=torch.bool)
        
        # Clinical features (use latest visit)
        latest_visit = visits[-1]
        clinical_features = torch.tensor([
            latest_visit['metadata']['age'],
            latest_visit['metadata']['breast_area'],
            latest_visit['metadata']['dense_area'],
            latest_visit['metadata']['percent_density']
        ], dtype=torch.float32)
        
        # Outcomes
        outcomes = sequence['outcomes']
        targets = torch.tensor([
            outcomes[f'cancer_{year}year'] for year in range(1, Config.PREDICTION_YEARS + 1)
        ], dtype=torch.float32)
        
        return {
            'images': images,
            'clinical_features': clinical_features,
            'visit_mask': visit_mask,
            'targets': targets,
            'patient_id': sequence['patient_id'],
            'sequence_id': seq_id
        }

def create_patient_splits(sequences: Dict, test_size: float = 0.2, 
                         val_size: float = 0.2, random_state: int = 42) -> Tuple[List, List, List]:
    """Create patient-level train/val/test splits."""
    
    # Get unique patient IDs
    patient_ids = list(set([seq_data['patient_id'] for seq_data in sequences.values()]))
    
    # Split patients
    train_patients, test_patients = train_test_split(
        patient_ids, test_size=test_size, random_state=random_state
    )
    
    train_patients, val_patients = train_test_split(
        train_patients, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Map sequences to splits
    train_sequences = [seq_id for seq_id, seq_data in sequences.items() 
                      if seq_data['patient_id'] in train_patients]
    val_sequences = [seq_id for seq_id, seq_data in sequences.items() 
                    if seq_data['patient_id'] in val_patients]
    test_sequences = [seq_id for seq_id, seq_data in sequences.items() 
                     if seq_data['patient_id'] in test_patients]
    
    logger.info(f"Patient splits: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")
    logger.info(f"Sequence splits: Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")
    
    return train_sequences, val_sequences, test_sequences

class SurvivalLoss(nn.Module):
    """Loss function for survival analysis with right-censoring."""
    
    def __init__(self, prediction_years: int = 5):
        super().__init__()
        self.prediction_years = prediction_years
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor):
        """
        Args:
            predictions: Dict with keys like 'risk_1year', 'risk_2year', etc.
            targets: (batch_size, prediction_years) - binary outcomes
        """
        total_loss = 0.0
        
        for year in range(1, self.prediction_years + 1):
            pred_key = f'risk_{year}year'
            if pred_key in predictions:
                pred = predictions[pred_key]
                target = targets[:, year - 1]
                
                # Apply BCELoss
                loss = self.bce_loss(pred, target)
                total_loss += loss
        
        return total_loss / self.prediction_years

def calculate_metrics(predictions: Dict[str, np.ndarray], 
                     targets: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Calculate evaluation metrics."""
    
    metrics = {}
    
    for year in range(1, Config.PREDICTION_YEARS + 1):
        pred_key = f'risk_{year}year'
        if pred_key in predictions:
            pred = predictions[pred_key]
            target = targets[:, year - 1]
            
            # ROC AUC
            try:
                auc = roc_auc_score(target, pred)
            except:
                auc = 0.5
            
            # C-index (same as AUC for binary classification)
            try:
                c_index = concordance_index(target, pred)
            except:
                c_index = 0.5
            
            metrics[f'{year}_year'] = {
                'auc': auc,
                'c_index': c_index
            }
    
    return metrics

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                metric_func, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence intervals."""
    
    bootstrap_scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue
    
    if not bootstrap_scores:
        return 0.0, 0.0
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return lower, upper

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Config) -> Tuple[nn.Module, Dict]:
    """Train the model."""
    
    model = model.to(config.DEVICE)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    # Loss function
    criterion = SurvivalLoss(config.PREDICTION_YEARS)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    
    logger.info("Starting training...")
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            images = batch['images'].to(config.DEVICE)
            clinical_features = batch['clinical_features'].to(config.DEVICE)
            visit_mask = batch['visit_mask'].to(config.DEVICE)
            targets = batch['targets'].to(config.DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            risks, hazards = model(images, clinical_features, visit_mask)
            
            # Calculate loss
            loss = criterion(risks, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = {f'risk_{year}year': [] for year in range(1, config.PREDICTION_YEARS + 1)}
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                images = batch['images'].to(config.DEVICE)
                clinical_features = batch['clinical_features'].to(config.DEVICE)
                visit_mask = batch['visit_mask'].to(config.DEVICE)
                targets = batch['targets'].to(config.DEVICE)
                
                # Forward pass
                risks, hazards = model(images, clinical_features, visit_mask)
                
                # Calculate loss
                loss = criterion(risks, targets)
                val_loss += loss.item()
                
                # Store predictions
                for year in range(1, config.PREDICTION_YEARS + 1):
                    pred_key = f'risk_{year}year'
                    val_predictions[pred_key].append(risks[pred_key].cpu().numpy())
                
                val_targets.append(targets.cpu().numpy())
        
        # Aggregate validation results
        for key in val_predictions:
            val_predictions[key] = np.concatenate(val_predictions[key])
        val_targets = np.concatenate(val_targets)
        
        # Calculate metrics
        val_metrics = calculate_metrics(val_predictions, val_targets)
        
        # Update history
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Print validation metrics
        for year_key, metrics in val_metrics.items():
            logger.info(f"  {year_key} - AUC: {metrics['auc']:.4f}, C-Index: {metrics['c_index']:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.info("Training completed!")
    return model, history

def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  config: Config) -> Tuple[Dict, Dict]:
    """Evaluate the model."""
    
    model.eval()
    model = model.to(config.DEVICE)
    
    # Collect predictions
    predictions = {f'risk_{year}year': [] for year in range(1, config.PREDICTION_YEARS + 1)}
    targets = []
    patient_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            images = batch['images'].to(config.DEVICE)
            clinical_features = batch['clinical_features'].to(config.DEVICE)
            visit_mask = batch['visit_mask'].to(config.DEVICE)
            batch_targets = batch['targets']
            
            # Forward pass
            risks, hazards = model(images, clinical_features, visit_mask)
            
            # Store predictions
            for year in range(1, config.PREDICTION_YEARS + 1):
                pred_key = f'risk_{year}year'
                predictions[pred_key].append(risks[pred_key].cpu().numpy())
            
            targets.append(batch_targets.numpy())
            patient_ids.extend(batch['patient_id'])
    
    # Aggregate results
    for key in predictions:
        predictions[key] = np.concatenate(predictions[key])
    targets = np.concatenate(targets)
    
    # Calculate metrics with confidence intervals
    detailed_metrics = {}
    
    for year in range(1, config.PREDICTION_YEARS + 1):
        pred_key = f'risk_{year}year'
        pred = predictions[pred_key]
        target = targets[:, year - 1]
        
        # AUC
        auc = roc_auc_score(target, pred)
        auc_ci = bootstrap_ci(target, pred, roc_auc_score)
        
        # C-index
        c_index = concordance_index(target, pred)
        c_index_ci = bootstrap_ci(target, pred, concordance_index)
        
        detailed_metrics[f'{year}_year'] = {
            'auc': auc,
            'auc_ci': auc_ci,
            'c_index': c_index,
            'c_index_ci': c_index_ci,
            'n_positive': int(target.sum()),
            'n_total': len(target)
        }
    
    results = {
        'predictions': predictions,
        'targets': targets,
        'patient_ids': patient_ids,
        'metrics': detailed_metrics
    }
    
    return results, detailed_metrics

def create_results_table(all_results: Dict[int, Dict]) -> pd.DataFrame:
    """Create results table for different history settings."""
    
    rows = []
    
    for history_years, results in all_results.items():
        metrics = results['metrics']
        
        row = {'History_Years': history_years}
        
        for year in range(1, Config.PREDICTION_YEARS + 1):
            year_key = f'{year}_year'
            if year_key in metrics:
                metric_data = metrics[year_key]
                
                # AUC
                auc = metric_data['auc']
                auc_ci = metric_data['auc_ci']
                row[f'AUC_{year}year'] = f"{auc:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})"
                
                # C-Index
                c_index = metric_data['c_index']
                c_index_ci = metric_data['c_index_ci']
                row[f'CIndex_{year}year'] = f"{c_index:.3f} ({c_index_ci[0]:.3f}-{c_index_ci[1]:.3f})"
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def plot_results(all_results: Dict[int, Dict], save_path: str = None):
    """Plot comparison results."""
    
    # Extract data for plotting
    history_years = sorted(all_results.keys())
    
    # Prepare data
    auc_data = {year: [] for year in range(1, Config.PREDICTION_YEARS + 1)}
    auc_ci_data = {year: [] for year in range(1, Config.PREDICTION_YEARS + 1)}
    
    for hist_year in history_years:
        metrics = all_results[hist_year]['metrics']
        
        for pred_year in range(1, Config.PREDICTION_YEARS + 1):
            year_key = f'{pred_year}_year'
            if year_key in metrics:
                auc_data[pred_year].append(metrics[year_key]['auc'])
                ci = metrics[year_key]['auc_ci']
                auc_ci_data[pred_year].append([ci[0], ci[1]])
            else:
                auc_data[pred_year].append(0.5)
                auc_ci_data[pred_year].append([0.5, 0.5])
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Vision Transformer Performance Across History Lengths', fontsize=16)
    
    # Individual year plots
    for pred_year in range(1, Config.PREDICTION_YEARS + 1):
        row = (pred_year - 1) // 3
        col = (pred_year - 1) % 3
        
        ax = axes[row, col]
        
        aucs = auc_data[pred_year]
        cis = np.array(auc_ci_data[pred_year])
        
        ax.plot(history_years, aucs, 'o-', linewidth=2, markersize=8, label=f'{pred_year}-year AUC')
        ax.fill_between(history_years, cis[:, 0], cis[:, 1], alpha=0.3)
        
        ax.set_xlabel('History Years')
        ax.set_ylabel('AUC')
        ax.set_title(f'{pred_year}-Year Cancer Risk Prediction')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)
        ax.legend()
    
    # Overall comparison plot
    ax = axes[1, 2]
    
    for pred_year in range(1, Config.PREDICTION_YEARS + 1):
        aucs = auc_data[pred_year]
        ax.plot(history_years, aucs, 'o-', linewidth=2, markersize=6, label=f'{pred_year}-year')
    
    ax.set_xlabel('History Years')
    ax.set_ylabel('AUC')
    ax.set_title('All Prediction Horizons')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to run the complete analysis."""
    
    print("="*80)
    print("VISION TRANSFORMER FOR BREAST CANCER RISK PREDICTION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Set random seeds
    set_random_seeds(Config.SEED)
    
    # Initialize config
    config = Config()
    
    # Initialize data processor
    data_processor = DataProcessor(config.METADATA_PATH, config.IMAGES_PATH)
    
    # Load metadata and image paths
    logger.info("Loading metadata and image paths...")
    df = data_processor.load_metadata()
    image_paths = data_processor.find_image_paths()
    
    # Store results for all history settings
    all_results = {}
    
    # Run analysis for different history settings
    for history_years in range(0, config.MAX_HISTORY_YEARS + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING ANALYSIS WITH {history_years} YEARS OF HISTORY")
        logger.info(f"{'='*80}")
        
        # Create sequences
        sequences = data_processor.create_visit_sequences(df, history_years)
        
        if not sequences:
            logger.warning(f"No valid sequences for history_years={history_years}")
            continue
        
        # Create patient-level splits
        train_seq_ids, val_seq_ids, test_seq_ids = create_patient_splits(sequences)
        
        # Filter sequences by split
        train_sequences = {seq_id: sequences[seq_id] for seq_id in train_seq_ids if seq_id in sequences}
        val_sequences = {seq_id: sequences[seq_id] for seq_id in val_seq_ids if seq_id in sequences}
        test_sequences = {seq_id: sequences[seq_id] for seq_id in test_seq_ids if seq_id in sequences}
        
        # Create datasets
        train_dataset = MammographyDataset(train_sequences, image_paths, history_years, max_visits=history_years+1)
        val_dataset = MammographyDataset(val_sequences, image_paths, history_years, max_visits=history_years+1)
        test_dataset = MammographyDataset(test_sequences, image_paths, history_years, max_visits=history_years+1)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                                num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                              num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                               num_workers=2, pin_memory=True)
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        # Initialize model
        model = BreastCancerVisionTransformer(config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters")
        
        # Train model
        logger.info("Training model...")
        trained_model, history = train_model(model, train_loader, val_loader, config)
        
        # Evaluate model
        logger.info("Evaluating model...")
        results, detailed_metrics = evaluate_model(trained_model, test_loader, config)
        
        # Store results
        all_results[history_years] = {
            'results': results,
            'metrics': detailed_metrics,
            'history': history,
            'model_params': total_params
        }
        
        # Print results summary
        logger.info(f"\nResults for {history_years} years of history:")
        for year_key, metrics in detailed_metrics.items():
            auc = metrics['auc']
            auc_ci = metrics['auc_ci']
            c_index = metrics['c_index']
            c_index_ci = metrics['c_index_ci']
            
            logger.info(f"  {year_key}: AUC = {auc:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f}), "
                       f"C-Index = {c_index:.3f} ({c_index_ci[0]:.3f}-{c_index_ci[1]:.3f})")
    
    # Create comprehensive comparison
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE COMPARISON ANALYSIS")
    logger.info(f"{'='*80}")
    
    # Create results table
    results_table = create_results_table(all_results)
    
    # Print results table
    print("\nRESULTS SUMMARY TABLE")
    print("-" * 80)
    print(results_table.to_string(index=False))
    
    # Save results
    results_table.to_csv('vision_transformer_results_summary.csv', index=False)
    
    # Save detailed results
    detailed_results = {}
    for history_years, data in all_results.items():
        detailed_results[history_years] = {
            'metrics': data['metrics'],
            'model_params': data['model_params']
        }
    
    with open('vision_transformer_detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    # Create plots
    plot_results(all_results, save_path='vision_transformer_results_plot.png')
    
    # Final analysis
    print(f"\n{'='*50}")
    print("FINAL ANSWER TO RESEARCH QUESTION:")
    print(f"{'='*50}")
    
    # Find best performance
    best_history = None
    best_auc = 0.0
    
    for history_years, data in all_results.items():
        # Use 3-year prediction as primary metric
        if '3_year' in data['metrics']:
            auc = data['metrics']['3_year']['auc']
            if auc > best_auc:
                best_auc = auc
                best_history = history_years
    
    if best_history is not None:
        baseline_auc = all_results[0]['metrics']['3_year']['auc'] if '3_year' in all_results[0]['metrics'] else 0.5
        improvement = best_auc - baseline_auc
        
        if improvement > 0.02:  # Significant improvement threshold
            print(f"✅ YES - More history IS better!")
            print(f"   3-year AUC improved from {baseline_auc:.3f} to {best_auc:.3f}")
            print(f"   Improvement: +{improvement:.3f}")
            print(f"🏆 BEST CONFIGURATION: {best_history} years (AUC = {best_auc:.3f})")
        else:
            print(f"❌ NO - More history does not significantly improve performance")
            print(f"   Maximum improvement: +{improvement:.3f} (threshold: 0.02)")
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Complete analysis finished!")

if __name__ == "__main__":
    main()