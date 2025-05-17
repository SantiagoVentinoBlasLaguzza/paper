import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# MODIFICATION: Import GroupShuffleSplit
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from collections import Counter # For class weights
import matplotlib.pyplot as plt
import warnings
import logging
# MODIFICATION: Import for t-SNE
from sklearn.manifold import TSNE
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# --- Configuration & Seeding (Point 4.2) ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Mixed Precision Setting (Point 3.1)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

PT_FILES_DIR = "matrix"
OUTPUT_MODEL_DIR = "cvae_pytorch_models_AD_CN_classifier_phd_v5_leakage_fix" # MODIFICATION: Changed version
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Logging Setup (Point 4.3)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(OUTPUT_MODEL_DIR, "training_log_phd_v5.txt")), # MODIFICATION: Changed version
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

# Model Hyperparameters
IMG_CHANNELS = 3
IMG_RES = 116 # Point 4.1
LATENT_DIM = 128 * 2  # 256
GN_NUM_GROUPS = 32 # NEW: Number of groups for GroupNorm

# --- ResNet-style Encoder Parameters ---
INITIAL_CONV_FILTERS = 64
INITIAL_KERNEL_SIZE = 7
INITIAL_STRIDE = 2
INITIAL_PADDING = 3
INITIAL_MAX_POOL = True
RESNET_STAGES_PARAMS = [
    (2, [64, 64, 256], 1),
    (2, [128, 128, 512], 2),
    (2, [256, 256, 1024], 2),
]
RESNET_KERNEL_SIZE = 3
ENCODER_DENSE_UNITS = [512]

# --- Decoder Parameters ---
DECODER_FC_TARGET_H_W = 8
DECODER_START_CONV_CHANNELS = RESNET_STAGES_PARAMS[-1][1][2]
DECODER_CONVTRANSPOSE_LAYERS_PARAMS = [
    (512, 3, 2, 1, 0),
    (256, 3, 2, 1, 0),
    (128, 4, 2, 1, 0),
]
FINAL_DECODER_CONVTRANSPOSE_PARAMS = (IMG_CHANNELS, 4, 2, 1, 0)
DECODER_DENSE_UNITS = [512]

# Training Hyperparameters
CVAE_EPOCHS = 400
CVAE_BATCH_SIZE = 32
CVAE_LEARNING_RATE = 1e-4
CVAE_WEIGHT_DECAY = 1e-5

# Beta VAE / Capacity Control (Point 3.2 & 3.3)
BETA_START = 0.001
BETA_END = 1.0
BETA_ANNEAL_EPOCHS = 100
C_MAX = 25.0
C_ANNEAL_EPOCHS_CAPACITY = int(BETA_ANNEAL_EPOCHS * 0.8)

# Classifier Hyperparameters
CLASSIFIER_DENSE_UNITS = [32]
CLASSIFIER_DROPOUT_RATE = 0.2
CLASSIFIER_EPOCHS = 200
CLASSIFIER_LR = 1e-4
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_BATCH_SIZE = 16
NUM_BINARY_CLASSES = 2

EARLY_STOPPING_PATIENCE_CVAE = 35
EARLY_STOPPING_PATIENCE_CLF = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
USE_AMP = torch.cuda.is_available()

# --- EarlyStopping Class ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=logger.info, mode='min'):
        self.patience = patience; self.verbose = verbose; self.counter = 0
        self.best_score = None; self.early_stop = False
        self.val_metric_best = np.Inf if mode == 'min' else -np.Inf
        self.delta = delta; self.path = path; self.trace_func = trace_func
        self.mode = mode

    def __call__(self, current_val_metric, model):
        score = -current_val_metric if self.mode == 'min' else current_val_metric
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(current_val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self.save_checkpoint(current_val_metric, model); self.counter = 0

    def save_checkpoint(self, current_val_metric, model):
        if self.verbose:
            log_msg = f'Validation metric improved ({self.val_metric_best:.6f} --> {current_val_metric:.6f}). Saving model ...' if self.mode == 'max' else \
                      f'Validation loss decreased ({self.val_metric_best:.6f} --> {current_val_metric:.6f}). Saving model ...'
            self.trace_func(log_msg)
        torch.save(model.state_dict(), self.path)
        self.val_metric_best = current_val_metric

# --- 1. Data Loading and Preprocessing ---
class ConnectomeDataset(Dataset):
    def __init__(self, pt_file_paths_override, selected_metrics, pathology_mapping,
                 condition_preprocessors_input=None, data_scalers_input=None,
                 fit_scalers=True):
        self.pt_files = pt_file_paths_override
        self.selected_metrics = selected_metrics
        self.pathology_mapping = pathology_mapping
        self.pathology_categories = ["AD", "CN", "Other"]
        self.matrices, self.subject_ids, self.ages_raw, self.sexes_raw, self.pathologies_mapped_raw = [], [], [], [], []

        if not self.pt_files: # Handle empty list of files
            logger.warning("ConnectomeDataset initialized with no pt_file_paths_override. Dataset will be empty.")
            # Initialize attributes to prevent errors later if len() is called
            self.matrices = torch.empty(0) 
            self.conditions_for_cvae = torch.empty(0)
            self.pathology_labels_str = []
            self.subject_ids = [] # Ensure subject_ids is initialized for overlap check
            return


        for file_path in self.pt_files:
            try:
                content = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
                meta = content["meta"]; data_tensor_all_metrics = content["data"]
                original_group = meta.get("Group"); mapped_group = self.pathology_mapping.get(original_group)
                
                subject_id_current_file = meta.get("SubjectID")
                if subject_id_current_file is None:
                    logger.warning(f"File {file_path} is missing SubjectID. Assigning placeholder or consider skipping.")
                    subject_id_current_file = f"UNKNOWN_SID_FOR_{os.path.basename(file_path)}"

                if mapped_group is None: 
                    logger.warning(f"File {file_path} with group {original_group} cannot be mapped. Skipping this file.")
                    continue # Skip if pathology cannot be mapped

                metric_indices = [meta["MetricsOrder"].index(m) for m in self.selected_metrics]
                self.matrices.append(data_tensor_all_metrics[metric_indices, :, :])
                self.ages_raw.append(float(meta.get("Age", np.nan)))
                sex_raw = meta.get("Sex", "Unknown")
                if sex_raw in ("M", "Male"): sex_clean = "Male"
                elif sex_raw in ("F", "Female"): sex_clean = "Female"
                else: sex_clean = "Unknown"
                self.sexes_raw.append(sex_clean)
                self.pathologies_mapped_raw.append(mapped_group)
                self.subject_ids.append(subject_id_current_file) 
            except Exception as e: logger.error(f"Skipping file {file_path} due to error during loading/meta extraction: {e}")

        if not self.matrices: 
            logger.warning("No data loaded into ConnectomeDataset after processing files (all might have been skipped or paths were empty).")
            self.matrices = torch.empty(0)
            self.conditions_for_cvae = torch.empty(0)
            self.pathology_labels_str = []
            # self.subject_ids is already an empty list if matrices is empty
            return

        self.matrices = torch.stack(self.matrices).float()
        ages_np_temp = np.array(self.ages_raw).astype(float)
        
        # Handle age imputation mean calculation
        if fit_scalers:
            # Calculate mean_age only if there are non-NaN ages
            valid_ages = ages_np_temp[~np.isnan(ages_np_temp)]
            self.mean_age = np.mean(valid_ages) if len(valid_ages) > 0 else 70.0 # Default if all are NaN
        
        # Use the calculated or provided mean for imputation
        mean_age_to_use = self.mean_age if fit_scalers else (condition_preprocessors_input.get('age_mean', 70.0) if condition_preprocessors_input else 70.0)
        ages_np = np.nan_to_num(ages_np_temp, nan=mean_age_to_use).reshape(-1,1)
        
        sexes_np = np.array(self.sexes_raw).reshape(-1, 1)
        pathologies_ohe_np = np.array(self.pathologies_mapped_raw).reshape(-1,1)

        if fit_scalers:
            self.age_scaler = MinMaxScaler(feature_range=(0,1))
            self.sex_encoder = OneHotEncoder(categories=[['Female', 'Male', 'Unknown']], sparse_output=False, handle_unknown='ignore') # handle_unknown='ignore' or 'error'
            self.pathology_encoder_ohe_for_cvae = OneHotEncoder(categories=[self.pathology_categories], sparse_output=False, handle_unknown='ignore')
            
            self.ages_scaled = self.age_scaler.fit_transform(ages_np)
            self.sexes_encoded = self.sex_encoder.fit_transform(sexes_np)
            self.pathologies_encoded_for_cvae = self.pathology_encoder_ohe_for_cvae.fit_transform(pathologies_ohe_np)
            self.condition_preprocessors = {
                'age_scaler': self.age_scaler,
                'sex_encoder': self.sex_encoder,
                'pathology_ohe_encoder_for_cvae': self.pathology_encoder_ohe_for_cvae,
                'age_mean': self.mean_age
            }
        else:
            self.condition_preprocessors = condition_preprocessors_input
            self.ages_scaled = self.condition_preprocessors['age_scaler'].transform(ages_np)
            self.sexes_encoded = self.condition_preprocessors['sex_encoder'].transform(sexes_np)
            self.pathologies_encoded_for_cvae = self.condition_preprocessors['pathology_ohe_encoder_for_cvae'].transform(pathologies_ohe_np)

        self.conditions_for_cvae = torch.tensor(np.concatenate(
            [self.ages_scaled, self.sexes_encoded, self.pathologies_encoded_for_cvae], axis=1), dtype=torch.float32)
        
        self.data_scalers = [] if data_scalers_input is None else data_scalers_input
        for i in range(self.matrices.shape[1]): # Iterate over channels
            channel_data = self.matrices[:, i, :, :].reshape(self.matrices.shape[0], -1)
            if fit_scalers:
                if self.selected_metrics[i] == "GrangerCausality_Directed_FDR": channel_data = torch.log1p(channel_data)
                scaler = MinMaxScaler(feature_range=(0,1))
                self.matrices[:, i, :, :] = torch.tensor(scaler.fit_transform(channel_data.numpy()), dtype=torch.float32).reshape(self.matrices.shape[0], IMG_RES, IMG_RES)
                self.data_scalers.append(scaler)
            else:
                scaler = self.data_scalers[i]
                if self.selected_metrics[i] == "GrangerCausality_Directed_FDR": channel_data = torch.log1p(channel_data)
                self.matrices[:, i, :, :] = torch.tensor(scaler.transform(channel_data.numpy()), dtype=torch.float32).reshape(self.matrices.shape[0], IMG_RES, IMG_RES)
        self.pathology_labels_str = self.pathologies_mapped_raw

    def __len__(self): return len(self.matrices) if isinstance(self.matrices, torch.Tensor) and self.matrices.nelement() > 0 else 0

    def __getitem__(self, idx):
        if self.__len__() == 0:
            raise IndexError("Dataset is empty")
        binary_label = 1 if self.pathology_labels_str[idx] == "AD" else (0 if self.pathology_labels_str[idx] == "CN" else -1) # -1 for 'Other' or unmapped
        return self.matrices[idx], self.conditions_for_cvae[idx], torch.tensor(binary_label, dtype=torch.long), self.pathology_labels_str[idx]

    def get_preprocessors(self): return self.condition_preprocessors, self.data_scalers
    def get_condition_dim(self): return self.conditions_for_cvae.shape[1] if self.__len__() > 0 else 0


def make_connectome_dataset(pt_file_paths, selected_metrics, pathology_mapping,
                            fit_scalers, cond_proc_input=None, data_scalers_input=None):
    logger.info(f"Creating dataset (fit_scalers={fit_scalers}) with {len(pt_file_paths)} files planned...")
    dataset = ConnectomeDataset(
        pt_file_paths_override=pt_file_paths,
        selected_metrics=selected_metrics,
        pathology_mapping=pathology_mapping,
        condition_preprocessors_input=cond_proc_input,
        data_scalers_input=data_scalers_input,
        fit_scalers=fit_scalers
    )
    logger.info(f"Actual dataset size after loading: {len(dataset)}")
    return dataset

# --- ResNet-style Blocks ---
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, num_groups_for_gn):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(num_groups_for_gn if filters1 > 0 and filters1 % num_groups_for_gn == 0 else max(1, filters1 // (filters1//num_groups_for_gn)) if filters1 > 0 and (filters1//num_groups_for_gn)>0 else 1 , filters1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.gn2 = nn.GroupNorm(num_groups_for_gn if filters2 > 0 and filters2 % num_groups_for_gn == 0 else max(1, filters2 // (filters2//num_groups_for_gn)) if filters2 > 0 and (filters2//num_groups_for_gn)>0 else 1, filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(num_groups_for_gn if filters3 > 0 and filters3 % num_groups_for_gn == 0 else max(1, filters3 // (filters3//num_groups_for_gn)) if filters3 > 0 and (filters3//num_groups_for_gn)>0 else 1, filters3)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        out += shortcut
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=2, num_groups_for_gn=32):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(num_groups_for_gn if filters1 > 0 and filters1 % num_groups_for_gn == 0 else max(1, filters1 // (filters1//num_groups_for_gn)) if filters1 > 0 and (filters1//num_groups_for_gn)>0 else 1, filters1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.gn2 = nn.GroupNorm(num_groups_for_gn if filters2 > 0 and filters2 % num_groups_for_gn == 0 else max(1, filters2 // (filters2//num_groups_for_gn)) if filters2 > 0 and (filters2//num_groups_for_gn)>0 else 1, filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(num_groups_for_gn if filters3 > 0 and filters3 % num_groups_for_gn == 0 else max(1, filters3 // (filters3//num_groups_for_gn)) if filters3 > 0 and (filters3//num_groups_for_gn)>0 else 1, filters3)

        self.shortcut_conv = nn.Conv2d(in_channels, filters3, kernel_size=1, stride=stride, padding=0, bias=False)
        self.shortcut_gn = nn.GroupNorm(num_groups_for_gn if filters3 > 0 and filters3 % num_groups_for_gn == 0 else max(1, filters3 // (filters3//num_groups_for_gn)) if filters3 > 0 and (filters3//num_groups_for_gn)>0 else 1, filters3)

    def forward(self, x):
        shortcut = self.shortcut_gn(self.shortcut_conv(x))
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        out += shortcut
        out = self.relu(out)
        return out

# --- CVAE Model Definition with ResNet-style Encoder ---
class CVAE_ResNetEncoder(nn.Module):
    def __init__(self, img_channels, img_res, condition_dim, latent_dim,
                 initial_conv_filters, initial_kernel_size, initial_stride, initial_padding, use_max_pool,
                 resnet_stages_params, resnet_kernel_size,
                 encoder_dense_units,
                 decoder_dense_units, decoder_start_conv_channels, decoder_fc_target_h_w,
                 decoder_convtranspose_params, final_decoder_convtranspose_params,
                 num_groups_for_gn):
        super(CVAE_ResNetEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder_start_conv_channels = decoder_start_conv_channels
        self.decoder_fc_target_h_w = decoder_fc_target_h_w
        self.num_groups_for_gn = num_groups_for_gn

        encoder_modules = []
        current_channels = img_channels
        
        def _get_gn(num_channels_gn_param):
            actual_num_groups = self.num_groups_for_gn
            if num_channels_gn_param <= 0: return nn.Identity() # Should not happen with valid channels
            if num_channels_gn_param < self.num_groups_for_gn or num_channels_gn_param % self.num_groups_for_gn != 0:
                # Try to find a good divisor, or default to a smaller number of groups or 1
                if num_channels_gn_param < self.num_groups_for_gn and num_channels_gn_param > 0 :
                     actual_num_groups = num_channels_gn_param 
                else: # Try common divisors or default to 1 if no good common divisor found
                    common_divs = [g for g in [16,8,4,2,1] if num_channels_gn_param % g == 0 and g <= self.num_groups_for_gn]
                    actual_num_groups = common_divs[0] if common_divs else 1 
            return nn.GroupNorm(actual_num_groups, num_channels_gn_param)

        encoder_modules.extend([
            nn.Conv2d(current_channels, initial_conv_filters, initial_kernel_size, initial_stride, initial_padding, bias=False),
            _get_gn(initial_conv_filters),
            nn.SiLU(inplace=True)])
        current_channels = initial_conv_filters
        if use_max_pool:
            encoder_modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        for _, (num_blocks, filters_list, stride_for_first) in enumerate(resnet_stages_params):
            encoder_modules.append(ConvBlock(current_channels, filters_list, resnet_kernel_size, stride=stride_for_first, num_groups_for_gn=self.num_groups_for_gn))
            current_channels = filters_list[2]
            for _ in range(num_blocks - 1):
                encoder_modules.append(IdentityBlock(current_channels, filters_list, resnet_kernel_size, num_groups_for_gn=self.num_groups_for_gn))

        encoder_modules.append(nn.AdaptiveAvgPool2d((1,1)))
        self.encoder_conv = nn.Sequential(*encoder_modules)
        self.flattened_size_after_resnet = current_channels

        encoder_fc_layers_list = []
        fc_input_size = self.flattened_size_after_resnet + condition_dim
        for units in encoder_dense_units:
            encoder_fc_layers_list.extend([nn.Linear(fc_input_size, units), nn.SiLU(inplace=True)])
            fc_input_size = units
        self.encoder_fc = nn.Sequential(*encoder_fc_layers_list)
        self.fc_z_mean = nn.Linear(fc_input_size, latent_dim)
        self.fc_z_log_var = nn.Linear(fc_input_size, latent_dim)

        decoder_fc_layers_list_dec = []
        current_fc_decoder_input = latent_dim + condition_dim
        decoder_fc_output_size = self.decoder_start_conv_channels * self.decoder_fc_target_h_w * self.decoder_fc_target_h_w
        for units in decoder_dense_units:
            decoder_fc_layers_list_dec.extend([nn.Linear(current_fc_decoder_input, units), nn.SiLU(inplace=True)])
            current_fc_decoder_input = units
        decoder_fc_layers_list_dec.extend([nn.Linear(current_fc_decoder_input, decoder_fc_output_size), nn.SiLU(inplace=True)])
        self.decoder_fc = nn.Sequential(*decoder_fc_layers_list_dec)

        decoder_conv_t_layers_list = []
        current_channels_dec = self.decoder_start_conv_channels
        for _, (out_channels, kernel, stride, padding, out_pad) in enumerate(decoder_convtranspose_params):
            decoder_conv_t_layers_list.extend([
                nn.ConvTranspose2d(current_channels_dec, out_channels, kernel, stride, padding, output_padding=out_pad, bias=False),
                _get_gn(out_channels),
                nn.SiLU(inplace=True)])
            current_channels_dec = out_channels
        out_c, ks, s, p, op = final_decoder_convtranspose_params
        decoder_conv_t_layers_list.extend([
            nn.ConvTranspose2d(current_channels_dec, out_c, ks, s, p, output_padding=op), nn.Sigmoid()])
        self.decoder_conv_transpose = nn.Sequential(*decoder_conv_t_layers_list)

    def encode(self, x_img, x_cond):
        x = self.encoder_conv(x_img); x = x.view(x.size(0), -1)
        x_combined = torch.cat([x, x_cond], dim=1); x_fc = self.encoder_fc(x_combined)
        return self.fc_z_mean(x_fc), self.fc_z_log_var(x_fc)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var); eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z, x_cond):
        z_combined = torch.cat([z, x_cond], dim=1); x_fc = self.decoder_fc(z_combined)
        x = x_fc.view(x_fc.size(0), self.decoder_start_conv_channels, self.decoder_fc_target_h_w, self.decoder_fc_target_h_w)
        return self.decoder_conv_transpose(x)

    def forward(self, x_img, x_cond):
        z_mean, z_log_var = self.encode(x_img, x_cond)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z, x_cond), z_mean, z_log_var

# --- Loss Function for CVAE with Capacity Control ---
def cvae_loss_function_with_capacity(x_reconstructed, x_original, z_mean, z_log_var, current_beta_kl, current_capacity_c):
    recon_loss = nn.MSELoss(reduction='sum')(x_reconstructed, x_original) / x_original.size(0)
    kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x_original.size(0)
    capacity_regularized_kl_loss = current_beta_kl * torch.abs(kl_div - current_capacity_c)
    total_loss = recon_loss + capacity_regularized_kl_loss
    return total_loss, recon_loss, kl_div

# --- Training Loop for CVAE ---
def train_cvae_epoch(model, dataloader, optimizer, scaler, current_beta_kl, current_capacity_c, device):
    model.train(); train_loss_sum = 0; recon_loss_sum = 0; kl_loss_sum = 0
    if len(dataloader) == 0: return 0,0,0 # Handle empty dataloader
    for matrices, conditions, _, _ in dataloader:
        matrices, conditions = matrices.to(device), conditions.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            reconstructed, z_mean, z_log_var = model(matrices, conditions)
            loss, recon_l, kl_l = cvae_loss_function_with_capacity(reconstructed, matrices, z_mean, z_log_var, current_beta_kl, current_capacity_c)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        train_loss_sum += loss.item(); recon_loss_sum += recon_l.item(); kl_loss_sum += kl_l.item()
    return train_loss_sum/len(dataloader), recon_loss_sum/len(dataloader), kl_loss_sum/len(dataloader)

def validate_cvae_epoch(model, dataloader, current_beta_kl, current_capacity_c, device):
    model.eval(); val_loss_sum = 0; recon_loss_sum = 0; kl_loss_sum = 0
    if len(dataloader) == 0: return 0,0,0 # Handle empty dataloader
    with torch.no_grad():
        for matrices, conditions, _, _ in dataloader:
            matrices, conditions = matrices.to(device), conditions.to(device)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                reconstructed, z_mean, z_log_var = model(matrices, conditions)
                loss, recon_l, kl_l = cvae_loss_function_with_capacity(reconstructed, matrices, z_mean, z_log_var, current_beta_kl, current_capacity_c)
            val_loss_sum += loss.item(); recon_loss_sum += recon_l.item(); kl_loss_sum += kl_l.item()
    avg_val_loss = val_loss_sum/len(dataloader) if len(dataloader) > 0 else 0
    return avg_val_loss, recon_loss_sum/len(dataloader) if len(dataloader) > 0 else 0, kl_loss_sum/len(dataloader) if len(dataloader) > 0 else 0

# --- Classifier Model (PyTorch) for AD vs CN ---
class LatentSpaceBinaryClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, hidden_units_list, dropout_rate=0.5):
        super(LatentSpaceBinaryClassifier, self).__init__()
        layers_list = []
        input_feat = latent_dim
        for hidden_units in hidden_units_list:
            layers_list.extend([
                nn.Linear(input_feat, hidden_units),
                nn.BatchNorm1d(hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)])
            input_feat = hidden_units
        layers_list.append(nn.Linear(input_feat, num_classes))
        self.network = nn.Sequential(*layers_list)
    def forward(self, latent_vector): return self.network(latent_vector)

# --- Training Loop for Classifier ---
def train_binary_classifier_epoch_v2(classifier, dataloader, optimizer, scaler, criterion, device):
    classifier.train(); total_loss = 0; correct_predictions = 0; total_samples = 0
    if len(dataloader) == 0: return 0,0 # Handle empty dataloader
    for latent_z, binary_labels in dataloader:
        latent_z, binary_labels = latent_z.to(device), binary_labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = classifier(latent_z); loss = criterion(outputs, binary_labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total_loss += loss.item(); _, predicted = torch.max(outputs.data, 1)
        total_samples += binary_labels.size(0); correct_predictions += (predicted == binary_labels).sum().item()
    return total_loss/len(dataloader) if len(dataloader) > 0 else 0, correct_predictions/total_samples if total_samples > 0 else 0

def evaluate_binary_classifier(classifier, dataloader, criterion, device):
    classifier.eval(); total_loss = 0; correct_predictions = 0; total_samples = 0
    all_probs_positive_class, all_labels_list = [], []
    if len(dataloader) == 0: return 0,0,[],[],0.5 # Handle empty dataloader
    with torch.no_grad():
        for latent_z, binary_labels in dataloader:
            latent_z, binary_labels = latent_z.to(device), binary_labels.to(device)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = classifier(latent_z); loss = criterion(outputs, binary_labels)
            total_loss += loss.item();
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total_samples += binary_labels.size(0); correct_predictions += (predicted == binary_labels).sum().item()
            all_probs_positive_class.extend(probs[:, 1].cpu().numpy())
            all_labels_list.extend(binary_labels.cpu().numpy())
    avg_loss = total_loss/len(dataloader) if len(dataloader) > 0 else 0
    avg_acc = correct_predictions/total_samples if total_samples > 0 else 0
    
    # Handle AUC calculation for empty or single-class scenarios
    if total_samples == 0 or len(np.unique(all_labels_list)) < 2:
        auc_score = 0.5 # Default or undefined AUC
    else:
        auc_score = roc_auc_score(all_labels_list, all_probs_positive_class)
    return avg_loss, avg_acc, all_labels_list, all_probs_positive_class, auc_score

# --- Main Execution ---
if __name__ == "__main__":
    selected_metrics = ["Correlation_FisherZ", "NMI", "GrangerCausality_Directed_FDR"]
    pathology_mapping = {"AD": "AD", "CN": "CN", "MCI": "Other", "LMCI": "Other", "EMCI": "Other"}

    logger.info("--- Initializing Datasets and Performing Subject-Aware Splits ---")
    all_pt_files_initial_list = sorted(glob.glob(os.path.join(PT_FILES_DIR, "*.pt")))
    if not all_pt_files_initial_list: 
        logger.error("No .pt files found in PT_FILES_DIR. Exiting.")
        exit()

    # Extract Subject IDs and Pathologies for each file, robustly
    all_pt_files_full_list = []
    current_files_subject_ids = []
    current_files_pathologies = []

    for file_path in all_pt_files_initial_list:
        try:
            content = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
            meta = content["meta"]
            subject_id = meta.get("SubjectID")
            original_group = meta.get("Group")
            mapped_group = pathology_mapping.get(original_group)

            if subject_id is None:
                logger.warning(f"File {file_path} is missing SubjectID. Assigning placeholder: UNKNOWN_SID_{os.path.basename(file_path)}")
                subject_id = f"UNKNOWN_SID_{os.path.basename(file_path)}" # Ensures it's treated as a unique group if no SID
            
            if mapped_group is None:
                logger.warning(f"File {file_path} group '{original_group}' not in pathology_mapping. Assigning 'Unknown_Group_For_Split'.")
                mapped_group = "Unknown_Group_For_Split"

            all_pt_files_full_list.append(file_path)
            current_files_subject_ids.append(subject_id)
            current_files_pathologies.append(mapped_group)
        except Exception as e:
            logger.error(f"Error loading metadata from {file_path}: {e}. Skipping this file.")
            continue # Skip this file and do not add to lists

    if not all_pt_files_full_list:
        logger.error("No valid .pt files remaining after initial loading and filtering. Exiting.")
        exit()
    
    logger.info(f"Total valid files found: {len(all_pt_files_full_list)}")

    num_files = len(all_pt_files_full_list)
    file_indices_arr = np.arange(num_files)
    
    train_indices, val_indices, test_indices = [], [], []

    try:
        # Split 1: Train+Val vs. Test
        gss_tv_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        # .split(X, y, groups) -> X can be file_indices_arr, y can be current_files_pathologies, groups are current_files_subject_ids
        train_val_indices_tmp, test_indices_tmp = next(gss_tv_test.split(file_indices_arr, current_files_pathologies, current_files_subject_ids))
        
        if len(train_val_indices_tmp) == 0:
            raise ValueError("Train+Validation split resulted in zero samples. Cannot proceed.")
        if len(test_indices_tmp) == 0:
            logger.warning("Test split resulted in zero samples. This might be acceptable for some use cases but is unusual.")


        # Prepare for second split: Train vs. Val from the train_val_indices_tmp
        # The test_size for gss_train_val is relative to the input data (the train_val set from the first split)
        # Original code used 0.20 of the (train+val) set for validation.
        gss_train_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED) 

        train_val_subset_pathologies = [current_files_pathologies[i] for i in train_val_indices_tmp]
        train_val_subset_subject_ids = [current_files_subject_ids[i] for i in train_val_indices_tmp]
        train_val_subset_indices_arr = np.arange(len(train_val_indices_tmp)) # Indices *within* the train_val_indices_tmp

        train_indices_in_subset, val_indices_in_subset = next(gss_train_val.split(train_val_subset_indices_arr, train_val_subset_pathologies, train_val_subset_subject_ids))

        # Map back to original file indices from all_pt_files_full_list
        train_indices = [train_val_indices_tmp[i] for i in train_indices_in_subset]
        val_indices = [train_val_indices_tmp[i] for i in val_indices_in_subset]
        test_indices = test_indices_tmp 

        if not train_indices or not val_indices : # Test indices might be empty if test_size is small and few groups
             raise ValueError("Train or Validation split is empty after GroupShuffleSplit.")
        if not test_indices:
            logger.warning("Test split is empty after GroupShuffleSplit. This might be acceptable for some use cases but is unusual.")


        logger.info("Successfully performed group-based stratified splits.")
        logger.info(f"Split sizes: Train files: {len(train_indices)}, Val files: {len(val_indices)}, Test files: {len(test_indices)}")

    except ValueError as e:
        logger.error(f"GroupShuffleSplit failed: {e}. This can happen with small datasets/groups or if a split results in zero samples. Consider reviewing data or split proportions.")
        logger.info("Attempting fallback to original stratified split (not group-aware). THIS MAY INTRODUCE LEAKAGE if subjects have multiple files.")
        # Fallback to original method - THIS IS NOT IDEAL FOR PREVENTING SUBJECT LEAKAGE
        # but provides a way for the script to run if GroupShuffleSplit fails catastrophically.
        # The user should be heavily warned.
        indices_orig = list(range(len(all_pt_files_full_list))) # Use indices of the current valid files
        temp_pathologies_orig = current_files_pathologies # Use pathologies of the current valid files
        try:
            train_val_indices, test_indices = train_test_split(
                indices_orig, test_size=0.20, random_state=SEED, stratify=temp_pathologies_orig)
            temp_train_val_pathologies = [temp_pathologies_orig[i] for i in train_val_indices]
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=0.20, random_state=SEED, stratify=temp_train_val_pathologies)
            logger.info("Fallback to original stratified split successful (BUT NOT GROUP-AWARE).")
        except Exception as e_fallback:
            logger.error(f"Fallback split also failed: {e_fallback}. Exiting.")
            exit()


    train_pt_files = [all_pt_files_full_list[i] for i in train_indices]
    val_pt_files = [all_pt_files_full_list[i] for i in val_indices]
    test_pt_files = [all_pt_files_full_list[i] for i in test_indices]
    
    # Create datasets
    train_dataset = make_connectome_dataset(train_pt_files, selected_metrics, pathology_mapping, fit_scalers=True)
    if len(train_dataset) == 0:
        logger.error("Train dataset is empty after creation. Cannot proceed. Check file loading and splitting logic.")
        exit()
    CONDITION_DIM = train_dataset.get_condition_dim()
    condition_preprocessors, data_scalers = train_dataset.get_preprocessors()

    val_dataset = make_connectome_dataset(val_pt_files, selected_metrics, pathology_mapping,
                                          fit_scalers=False, cond_proc_input=condition_preprocessors, data_scalers_input=data_scalers)
    test_dataset_final_holdout = make_connectome_dataset(test_pt_files, selected_metrics, pathology_mapping,
                                                         fit_scalers=False, cond_proc_input=condition_preprocessors, data_scalers_input=data_scalers)

    # --- Subject Overlap Check Function ---
    def check_subject_overlap(d_train, d_val, d_test, d_train_name="Train", d_val_name="Val", d_test_name="Test"):
        # Helper to safely get subject IDs
        def get_sids(dataset, name):
            if hasattr(dataset, 'subject_ids') and dataset.subject_ids is not None and len(dataset.subject_ids) > 0:
                return set(dataset.subject_ids)
            logger.warning(f"{name} dataset is empty or missing subject_ids attribute. Cannot check overlap for it.")
            return set()

        train_ids = get_sids(d_train, d_train_name)
        val_ids = get_sids(d_val, d_val_name)
        test_ids = get_sids(d_test, d_test_name)
        
        logger.info(f"Number of unique subjects in {d_train_name}: {len(train_ids)}")
        logger.info(f"Number of unique subjects in {d_val_name}: {len(val_ids)}")
        logger.info(f"Number of unique subjects in {d_test_name}: {len(test_ids)}")

        tv_overlap = train_ids & val_ids
        tt_overlap = train_ids & test_ids
        vt_overlap = val_ids & test_ids

        overlap_found = False
        if len(tv_overlap) > 0:
            logger.warning(f"SUBJECT OVERLAP: {d_train_name}-{d_val_name} overlap: {len(tv_overlap)} subjects. IDs (max 10): {list(tv_overlap)[:10]}")
            overlap_found = True
        if len(tt_overlap) > 0:
            logger.warning(f"SUBJECT OVERLAP: {d_train_name}-{d_test_name} overlap: {len(tt_overlap)} subjects. IDs (max 10): {list(tt_overlap)[:10]}")
            overlap_found = True
        if len(vt_overlap) > 0:
            logger.warning(f"SUBJECT OVERLAP: {d_val_name}-{d_test_name} overlap: {len(vt_overlap)} subjects. IDs (max 10): {list(vt_overlap)[:10]}")
            overlap_found = True
        
        if not overlap_found:
            logger.info("No subject overlap detected between splits. Group-based splitting appears successful.")
        else:
            logger.error("CRITICAL: Subject overlap detected. This indicates data leakage. Review splitting logic.")


    logger.info("\n--- Checking for Subject Overlap in Datasets ---")
    check_subject_overlap(train_dataset, val_dataset, test_dataset_final_holdout)

    train_loader_cvae = DataLoader(train_dataset, batch_size=CVAE_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader_cvae = DataLoader(val_dataset, batch_size=CVAE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    logger.info(f"CVAE Train set size: {len(train_dataset)}, CVAE Validation set size: {len(val_dataset)}, Final Test set size: {len(test_dataset_final_holdout)}")

    if len(train_dataset) == 0:
        logger.error("CVAE training dataset is empty. Exiting.")
        exit()

    cvae_model = CVAE_ResNetEncoder(
        img_channels=IMG_CHANNELS, img_res=IMG_RES, condition_dim=CONDITION_DIM, latent_dim=LATENT_DIM,
        initial_conv_filters=INITIAL_CONV_FILTERS, initial_kernel_size=INITIAL_KERNEL_SIZE,
        initial_stride=INITIAL_STRIDE, initial_padding=INITIAL_PADDING, use_max_pool=INITIAL_MAX_POOL,
        resnet_stages_params=RESNET_STAGES_PARAMS, resnet_kernel_size=RESNET_KERNEL_SIZE,
        encoder_dense_units=ENCODER_DENSE_UNITS,
        decoder_dense_units=DECODER_DENSE_UNITS,
        decoder_start_conv_channels=DECODER_START_CONV_CHANNELS,
        decoder_fc_target_h_w=DECODER_FC_TARGET_H_W,
        decoder_convtranspose_params=DECODER_CONVTRANSPOSE_LAYERS_PARAMS,
        final_decoder_convtranspose_params=FINAL_DECODER_CONVTRANSPOSE_PARAMS,
        num_groups_for_gn=GN_NUM_GROUPS
    ).to(device=DEVICE, memory_format=torch.channels_last if USE_AMP else torch.contiguous_format)

    optimizer_cvae = optim.Adam(cvae_model.parameters(), lr=CVAE_LEARNING_RATE, weight_decay=CVAE_WEIGHT_DECAY)
    scheduler_cvae = CosineAnnealingWarmRestarts(optimizer_cvae, T_0=max(10, CVAE_EPOCHS // 10), T_mult=2, eta_min=1e-7, verbose=False)
    cvae_early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE_CVAE, verbose=True,
                                       path=os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_best_model.pth"), mode='min')
    cvae_grad_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    logger.info("\n--- Training CVAE with ResNet Encoder (on AD, CN, Other) ---")
    cvae_train_losses_log, cvae_val_losses_log = [], []
    for epoch in range(1, CVAE_EPOCHS + 1):
        current_beta_kl = BETA_END / (1 + np.exp(-6 * (epoch / BETA_ANNEAL_EPOCHS - 0.5))) if BETA_ANNEAL_EPOCHS > 0 else BETA_END
        current_capacity_c = min(C_MAX, C_MAX * (epoch / C_ANNEAL_EPOCHS_CAPACITY) ) if C_ANNEAL_EPOCHS_CAPACITY > 0 else C_MAX
        
        train_loss, train_recon, train_kl = train_cvae_epoch(cvae_model, train_loader_cvae, optimizer_cvae, cvae_grad_scaler, current_beta_kl, current_capacity_c, DEVICE)
        val_loss, val_recon, val_kl = validate_cvae_epoch(cvae_model, val_loader_cvae, current_beta_kl, current_capacity_c, DEVICE) 
        scheduler_cvae.step() 

        cvae_train_losses_log.append(train_loss); cvae_val_losses_log.append(val_loss)
        if epoch % 10 == 0 or epoch == CVAE_EPOCHS or epoch == 1:
            logger.info(f"CVAE Epoch {epoch}/{CVAE_EPOCHS}: Beta_KL: {current_beta_kl:.4f} Capacity_C: {current_capacity_c:.2f} LR: {optimizer_cvae.param_groups[0]['lr']:.1e}\n"
                  f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})\n"
                  f"  Val Loss  : {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        
        if len(val_loader_cvae) > 0: # Only call early stopper if validation set is not empty
            cvae_early_stopper(val_loss, cvae_model)
            if cvae_early_stopper.early_stop: logger.info("CVAE Early stopping triggered."); break
        elif epoch >= EARLY_STOPPING_PATIENCE_CVAE : # If no val set, stop after patience epochs to prevent infinite run
             logger.info(f"CVAE: No validation set. Stopping after {epoch} epochs (patience equivalent).")
             # Save current model as "best" if no validation
             torch.save(cvae_model.state_dict(), os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_best_model.pth"))
             break
    
    logger.info("Loading best CVAE model weights.")
    # Ensure the best model path exists, especially if early stopping didn't trigger or val set was empty
    best_cvae_model_path = os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_best_model.pth")
    if os.path.exists(best_cvae_model_path):
        cvae_model.load_state_dict(torch.load(best_cvae_model_path))
    else:
        logger.warning(f"Best CVAE model path {best_cvae_model_path} not found. Using last trained model state.")
        # Optionally save the current state if no "best" was ever saved
        torch.save(cvae_model.state_dict(), best_cvae_model_path)


    plt.figure(figsize=(10,5)); plt.plot(cvae_train_losses_log, label="Train CVAE Loss"); 
    if cvae_val_losses_log and any(cvae_val_losses_log) : plt.plot(cvae_val_losses_log, label="Val CVAE Loss") # Plot only if val losses exist
    if cvae_val_losses_log and any(cvae_val_losses_log) and len(cvae_val_losses_log) > 0: 
        min_val_loss_epoch = np.argmin(cvae_val_losses_log) if any(v is not None and not np.isnan(v) for v in cvae_val_losses_log) else -1
        if min_val_loss_epoch != -1:
            plt.axvline(min_val_loss_epoch, color='r', linestyle='--', label=f'Best Val Epoch: {min_val_loss_epoch+1}')
    plt.title("CVAE ResNet Training & Validation Loss"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_loss_plot.png")); plt.close()


    logger.info("\n--- Preparing Data for AD vs CN Classifier ---")
    cvae_model.eval() 
    # MODIFICATION: extract_latent_and_filter now also returns subject_ids
    def extract_latent_and_filter(dataset_subset):
        latent_vectors_list, binary_labels_list, subject_ids_for_clf_list = [], [], []
        if len(dataset_subset) == 0: # Handle empty dataset_subset
            return None, None, None
            
        for i in range(len(dataset_subset)): 
            try:
                matrix, condition, binary_label, pathology_str = dataset_subset[i] 
                current_subject_id = dataset_subset.subject_ids[i] # Get subject_id for this sample
            except IndexError:
                logger.error(f"IndexError while accessing sample {i} or subject_id from dataset_subset. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Error getting sample {i} from dataset_subset: {e}. Skipping.")
                continue

            if pathology_str in ["AD", "CN"]:
                matrix, condition = matrix.unsqueeze(0).to(DEVICE), condition.unsqueeze(0).to(DEVICE)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    with torch.no_grad(): z_mean, _ = cvae_model.encode(matrix, condition)
                latent_vectors_list.append(z_mean.squeeze(0).cpu())
                binary_labels_list.append(binary_label.cpu())
                subject_ids_for_clf_list.append(current_subject_id) # Store subject_id
        
        if not latent_vectors_list: return None, None, None
        return torch.stack(latent_vectors_list), torch.tensor(binary_labels_list, dtype=torch.long), subject_ids_for_clf_list

    train_latent_clf, train_labels_clf, train_subject_ids_clf = extract_latent_and_filter(train_dataset)
    val_latent_clf, val_labels_clf, _ = extract_latent_and_filter(val_dataset) 
    test_latent_clf, test_labels_clf, _ = extract_latent_and_filter(test_dataset_final_holdout)

    if train_latent_clf is None or val_latent_clf is None: # Test can be None if no AD/CN samples
        logger.error("Insufficient AD/CN data in train or validation splits for classifier. Exiting."); exit()
    if test_latent_clf is None:
        logger.warning("No AD/CN samples found in the test set for classifier evaluation.")


    logger.info(f"AD/CN Classifier: Train size={len(train_latent_clf) if train_latent_clf is not None else 0}, Val size={len(val_latent_clf) if val_latent_clf is not None else 0}, Test size={len(test_latent_clf) if test_latent_clf is not None else 0}")
    if train_labels_clf is not None: logger.info(f"Train AD/CN dist: CN={(train_labels_clf==0).sum().item()}, AD={(train_labels_clf==1).sum().item()}")
    if val_labels_clf is not None: logger.info(f"Val AD/CN dist: CN={(val_labels_clf==0).sum().item()}, AD={(val_labels_clf==1).sum().item()}")
    if test_labels_clf is not None: logger.info(f"Test AD/CN dist: CN={(test_labels_clf==0).sum().item()}, AD={(test_labels_clf==1).sum().item()}")

    # Create TensorDatasets only if data is available
    train_dataset_clf_final = TensorDataset(train_latent_clf, train_labels_clf) if train_latent_clf is not None else None
    val_dataset_clf_final = TensorDataset(val_latent_clf, val_labels_clf) if val_latent_clf is not None else None
    test_dataset_clf_final_eval = TensorDataset(test_latent_clf, test_labels_clf) if test_latent_clf is not None else None

    train_loader_clf = DataLoader(train_dataset_clf_final, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True) if train_dataset_clf_final else None
    val_loader_clf = DataLoader(val_dataset_clf_final, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False) if val_dataset_clf_final else None
    test_loader_clf_final = DataLoader(test_dataset_clf_final_eval, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False) if test_dataset_clf_final_eval else None
    
    if not train_loader_clf:
        logger.error("Classifier training loader is empty. Cannot train classifier. Exiting.")
        exit()

    binary_classifier = LatentSpaceBinaryClassifier(
        latent_dim=LATENT_DIM, num_classes=NUM_BINARY_CLASSES,
        hidden_units_list=CLASSIFIER_DENSE_UNITS,
        dropout_rate=CLASSIFIER_DROPOUT_RATE
    ).to(DEVICE)

    if train_labels_clf is not None and len(train_labels_clf) > 0 :
        counts = Counter(train_labels_clf.tolist())
        cn_count = counts.get(0, 0); ad_count = counts.get(1, 0)
        if cn_count > 0 and ad_count > 0:
            weight_cn = (cn_count + ad_count) / (2.0 * cn_count)
            weight_ad = (cn_count + ad_count) / (2.0 * ad_count)
            class_weights_clf = torch.tensor([weight_cn, weight_ad], dtype=torch.float).to(DEVICE)
            criterion_binary_clf = nn.CrossEntropyLoss(weight=class_weights_clf)
            logger.info(f"Using class weights for classifier: CN={weight_cn:.2f}, AD={weight_ad:.2f}")
        else:
            criterion_binary_clf = nn.CrossEntropyLoss()
            logger.info("Using unweighted CrossEntropyLoss for classifier (one class missing in train_labels_clf or empty).")
    else:
        criterion_binary_clf = nn.CrossEntropyLoss()
        logger.info("Using unweighted CrossEntropyLoss for classifier (train_labels_clf is empty or None).")

    optimizer_binary_clf = optim.Adam(binary_classifier.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WEIGHT_DECAY)
    scheduler_clf = ReduceLROnPlateau(optimizer_binary_clf, mode='max', factor=0.1, patience=10, verbose=True)

    classifier_early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE_CLF, verbose=True,
                                             path=os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_best.pth"),
                                             mode='max') # Maximize validation AUC
    clf_grad_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    logger.info("\n--- Training AD vs CN Classifier on CVAE Latent Space ---")
    clf_train_losses_log, clf_train_accs_log, clf_val_losses_log, clf_val_accs_log, clf_val_auc_log = [], [], [], [], []
    
    best_clf_model_path = os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_best.pth")

    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        train_loss_clf, train_acc_clf = train_binary_classifier_epoch_v2(binary_classifier, train_loader_clf, optimizer_binary_clf, clf_grad_scaler, criterion_binary_clf, DEVICE)
        
        val_auc_clf = 0.5 # Default if no val loader
        if val_loader_clf and len(val_loader_clf) > 0:
            val_loss_clf, val_acc_clf, _, _, val_auc_clf = evaluate_binary_classifier(binary_classifier, val_loader_clf, criterion_binary_clf, DEVICE)
            scheduler_clf.step(val_auc_clf)
            classifier_early_stopper(val_auc_clf, binary_classifier)
        else: # No validation loader
            val_loss_clf, val_acc_clf = 0,0 
            logger.info(f"Classifier Epoch {epoch}/{CLASSIFIER_EPOCHS}: No validation set. LR: {optimizer_binary_clf.param_groups[0]['lr']:.1e}")
            if epoch >= EARLY_STOPPING_PATIENCE_CLF: # Stop after patience epochs if no validation
                logger.info(f"Classifier: No validation set. Stopping after {epoch} epochs (patience equivalent).")
                torch.save(binary_classifier.state_dict(), best_clf_model_path) # Save current as best
                break
        
        clf_train_losses_log.append(train_loss_clf); clf_train_accs_log.append(train_acc_clf)
        clf_val_losses_log.append(val_loss_clf); clf_val_accs_log.append(val_acc_clf); clf_val_auc_log.append(val_auc_clf)
        
        if epoch % 10 == 0 or epoch == CLASSIFIER_EPOCHS or epoch == 1:
            logger.info(f"Binary Classifier Epoch {epoch}/{CLASSIFIER_EPOCHS}: LR: {optimizer_binary_clf.param_groups[0]['lr']:.1e}\n"
                  f"  Train Loss: {train_loss_clf:.4f}, Train Acc: {train_acc_clf:.4f}\n"
                  f"  Val Loss  : {val_loss_clf:.4f}, Val Acc  : {val_acc_clf:.4f}, Val AUC: {val_auc_clf:.4f}")

        if val_loader_clf and len(val_loader_clf) > 0 and classifier_early_stopper.early_stop: 
            logger.info("Classifier Early stopping triggered."); break
    
    logger.info("Loading best binary classifier model weights.")
    if os.path.exists(best_clf_model_path):
        binary_classifier.load_state_dict(torch.load(best_clf_model_path))
    else:
        logger.warning(f"Best classifier model path {best_clf_model_path} not found. Using last trained model state.")
        torch.save(binary_classifier.state_dict(), best_clf_model_path) # Save current if no "best"
    
    # Plotting
    plt.figure(figsize=(18, 5)); 
    plt.subplot(1, 3, 1); plt.plot(clf_train_accs_log, label='Train Acc (AD/CN)'); 
    if any(clf_val_accs_log): plt.plot(clf_val_accs_log, label='Val Acc (AD/CN)')
    if clf_val_accs_log and any(v is not None and not np.isnan(v) for v in clf_val_accs_log) and len(clf_val_accs_log) > 0:
        max_val_acc_epoch = np.argmax(clf_val_accs_log) if any(clf_val_accs_log) else -1
        if max_val_acc_epoch != -1 : plt.axvline(max_val_acc_epoch , color='r', linestyle='--', label=f'Best Val Epoch (Acc): {max_val_acc_epoch+1 }')
    plt.title('Classifier Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 3, 2); plt.plot(clf_train_losses_log, label='Train Loss (AD/CN)'); 
    if any(clf_val_losses_log): plt.plot(clf_val_losses_log, label='Val Loss (AD/CN)')
    if clf_val_losses_log and any(v is not None and not np.isnan(v) for v in clf_val_losses_log) and len(clf_val_losses_log) > 0:
        min_val_loss_epoch = np.argmin(clf_val_losses_log) if any(clf_val_losses_log) else -1
        if min_val_loss_epoch != -1 : plt.axvline(min_val_loss_epoch, color='r', linestyle='--', label=f'Best Val Epoch (Loss): {min_val_loss_epoch+1}')
    plt.title('Classifier Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 3); 
    if any(clf_val_auc_log): plt.plot(clf_val_auc_log, label='Val AUC (AD/CN)')
    if clf_val_auc_log and any(v is not None and not np.isnan(v) for v in clf_val_auc_log) and len(clf_val_auc_log) > 0:
        max_val_auc_epoch = np.argmax(clf_val_auc_log) if any(clf_val_auc_log) else -1
        if max_val_auc_epoch != -1: plt.axvline(max_val_auc_epoch , color='r', linestyle='--', label=f'Best Val Epoch (AUC): {max_val_auc_epoch+1}')
    plt.title('Classifier Validation AUC'); plt.xlabel('Epochs'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
    
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_training_plot.png")); plt.close()

    # Final Evaluation on Test Set
    if test_loader_clf_final and len(test_loader_clf_final) > 0:
        logger.info("\n--- Final Evaluation on Test Set (AD vs CN Classifier) ---")
        test_loss_final, test_acc_final, true_test_labels, pred_test_probs_ad, test_auc_final = evaluate_binary_classifier(
            binary_classifier, test_loader_clf_final, criterion_binary_clf, DEVICE) 
        
        logger.info("\n--- Tuning Decision Threshold on Validation Set ---")
        best_threshold = 0.5 
        if val_loader_clf and len(val_loader_clf) > 0:
            _, _, val_true_labels_for_thresh, val_probs_ad_for_thresh, _ = evaluate_binary_classifier(
                binary_classifier, val_loader_clf, criterion_binary_clf, DEVICE)
            if len(val_true_labels_for_thresh) > 0 and len(np.unique(val_true_labels_for_thresh)) > 1 and len(val_probs_ad_for_thresh) > 0 : 
                fpr, tpr, thresholds = roc_curve(val_true_labels_for_thresh, val_probs_ad_for_thresh)
                if len(thresholds) > 0 : 
                    youden_j = tpr - fpr
                    if len(youden_j) > 0 : 
                        best_idx = np.argmax(youden_j)
                        best_threshold = thresholds[best_idx]
                        logger.info(f"Best decision threshold based on Youden's J on validation set: {best_threshold:.4f}")
                    else: logger.warning("Could not determine Youden's J (tpr-fpr array empty). Using 0.5.")
                else: logger.warning("Could not determine Youden's J (thresholds array empty). Using 0.5.")
            else: logger.warning("Not enough data or only one class present in validation set for threshold tuning. Using 0.5.")
        else:
            logger.warning("No validation data for threshold tuning. Using default 0.5.")

        pred_test_labels_for_report = (np.array(pred_test_probs_ad) >= best_threshold).astype(int)

        logger.info(f"Final Test Set Performance with default 0.5 threshold: Accuracy = {test_acc_final:.4f}, AUC = {test_auc_final:.4f}")
        if len(true_test_labels) > 0: # Only if there are test labels
            logger.info(f"Final Test Set Accuracy with tuned threshold ({best_threshold:.4f}): {(np.array(true_test_labels) == pred_test_labels_for_report).mean():.4f}")
            logger.info("\nFinal Test Set Classification Report (AD vs CN) with tuned threshold:")
            logger.info(f"\n{classification_report(true_test_labels, pred_test_labels_for_report, target_names=['CN (0)', 'AD (1)'], zero_division=0)}")
            logger.info("\nFinal Test Set Confusion Matrix (AD vs CN) with tuned threshold:")
            logger.info(f"\n{confusion_matrix(true_test_labels, pred_test_labels_for_report)}")
        else:
            logger.warning("No true labels in test set to generate report/matrix with tuned threshold.")
    else:
        logger.warning("Test loader is empty. Skipping final evaluation of classifier on test set.")


    # --- t-SNE Visualization of CVAE Latent Space (Train Data AD/CN) ---
    logger.info("\n--- Visualizing Latent Space with t-SNE (Train Data AD/CN) ---")
    if train_latent_clf is not None and len(train_latent_clf) > 0 and train_subject_ids_clf is not None:
        try:
            # Perplexity should be less than the number of samples.
            # n_components is the dimension of the embedded space.
            perplexity_val = min(30.0, float(len(train_latent_clf) - 1))
            if perplexity_val <=0: perplexity_val = 5.0 # Ensure perplexity is positive
            
            tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity_val, n_iter=1000, init='pca', learning_rate='auto')
            
            logger.info(f"Running t-SNE with perplexity: {perplexity_val} on {len(train_latent_clf)} samples.")
            latent_2d = tsne.fit_transform(train_latent_clf.cpu().numpy())
            
            plt.figure(figsize=(14, 10))
            
            # Determine if subject IDs should be used for style (avoid if too many unique subjects)
            unique_train_clf_subjects = list(set(train_subject_ids_clf))
            style_param = train_subject_ids_clf if len(unique_train_clf_subjects) <= 30 else None # Limit to 30 unique styles
            if style_param is None and len(unique_train_clf_subjects) > 0:
                 logger.info(f"Too many unique subjects ({len(unique_train_clf_subjects)}) in classifier training set for t-SNE style. Plotting without subject style.")
            elif not unique_train_clf_subjects:
                 style_param = None # No subject IDs available
                 logger.info("No subject IDs available for t-SNE style.")


            hue_labels = [f"{'AD' if l==1 else 'CN'}" for l in train_labels_clf.cpu().numpy()]

            sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], 
                            hue=hue_labels, 
                            style=style_param, 
                            palette="deep", s=70, alpha=0.7)

            plt.title('t-SNE of CVAE Latent Space (Train Data - AD/CN Filtered)')
            plt.xlabel('t-SNE Component 1'); plt.ylabel('t-SNE Component 2')
            # Adjust legend position if it's too crowded
            num_legend_items = len(set(hue_labels)) + (len(unique_train_clf_subjects) if style_param else 0)
            if num_legend_items > 10:
                plt.legend(title='Pathology / Subject', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make space for legend outside
            else:
                plt.legend(title='Pathology / Subject')
                plt.tight_layout()


            tsne_plot_path = os.path.join(OUTPUT_MODEL_DIR, "cvae_latent_tsne_plot.png")
            plt.savefig(tsne_plot_path)
            plt.close()
            logger.info(f"t-SNE plot saved to {tsne_plot_path}")

        except ImportError:
            logger.warning("seaborn or sklearn.manifold.TSNE not installed. Skipping t-SNE plot.")
        except Exception as e:
            logger.error(f"Error during t-SNE visualization: {e}", exc_info=True)
    else:
        logger.info("Training latent features (train_latent_clf) or their subject IDs not available/empty. Skipping t-SNE plot.")


    # --- Sanity Check: Logistic Regression on Latent Features ---
    logger.info("\n--- Sanity Check: Logistic Regression on Latent Features ---")
    from sklearn.linear_model import LogisticRegression 
    try:
        if train_latent_clf is not None and train_labels_clf is not None and len(train_latent_clf) > 0 and len(np.unique(train_labels_clf.cpu().numpy())) > 1:
            logreg = LogisticRegression(max_iter=4000, class_weight='balanced', random_state=SEED, solver='liblinear')
            logreg.fit(train_latent_clf.cpu().numpy(), train_labels_clf.cpu().numpy())
            
            if val_latent_clf is not None and val_labels_clf is not None and len(val_latent_clf) > 0 and len(np.unique(val_labels_clf.cpu().numpy())) > 1:
                val_probs_logreg = logreg.predict_proba(val_latent_clf.cpu().numpy())[:, 1]
                val_auc_logreg = roc_auc_score(val_labels_clf.cpu().numpy(), val_probs_logreg)
                logger.info(f"Logistic Regression - Validation AUC: {val_auc_logreg:.4f}")
            else: logger.warning("Skipping Logistic Regression validation AUC: Insufficient/invalid samples or only one class in validation set.")

            if test_latent_clf is not None and test_labels_clf is not None and len(test_latent_clf) > 0 and len(np.unique(test_labels_clf.cpu().numpy())) > 1:
                test_probs_logreg = logreg.predict_proba(test_latent_clf.cpu().numpy())[:, 1]
                test_auc_logreg = roc_auc_score(test_labels_clf.cpu().numpy(), test_probs_logreg)
                test_preds_logreg = (test_probs_logreg >= 0.5).astype(int) # Default 0.5 threshold for LR
                test_acc_logreg = (test_preds_logreg == test_labels_clf.cpu().numpy()).mean()
                logger.info(f"Logistic Regression - Test Accuracy: {test_acc_logreg:.4f}, Test AUC: {test_auc_logreg:.4f}")
                logger.info(f"Logistic Regression - Test Classification Report:\n{classification_report(test_labels_clf.cpu().numpy(), test_preds_logreg, target_names=['CN (0)', 'AD (1)'], zero_division=0)}")
            else: logger.warning("Skipping Logistic Regression test evaluation: Insufficient/invalid samples or only one class in test set.")
        else: logger.warning("Skipping Logistic Regression sanity check: Insufficient/invalid samples or only one class in training set.")
    except Exception as e: logger.error(f"Error during Logistic Regression sanity check: {e}", exc_info=True)

    logger.info("--- Script Finished ---")