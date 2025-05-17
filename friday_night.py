import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from collections import Counter # For class weights
import matplotlib.pyplot as plt
import warnings
import logging 

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
OUTPUT_MODEL_DIR = "cvae_pytorch_models_AD_CN_classifier_phd_v3" 
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Logging Setup (Point 4.3)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(OUTPUT_MODEL_DIR, "training_log_phd_v3.txt")),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

# Model Hyperparameters
IMG_CHANNELS = 3
IMG_RES = 116 # Point 4.1
LATENT_DIM = 128 * 2  # 256

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
    (512, 3, 2, 1, 0), # Output padding 1 for 8->15 ( (8-1)*2 - 2*1 + 3 + 1 = 16. (8-1)*2 -2*1 + 3 + 0 = 15)
    (256, 3, 2, 1, 0), # 15->29
    (128, 4, 2, 1, 0), # 29->58
]
FINAL_DECODER_CONVTRANSPOSE_PARAMS = (IMG_CHANNELS, 4, 2, 1, 0) # 58->116
DECODER_DENSE_UNITS = [512] 

# Training Hyperparameters
CVAE_EPOCHS = 400 # Adjusted based on typical early stopping
CVAE_BATCH_SIZE = 32
CVAE_LEARNING_RATE = 1e-4 
CVAE_WEIGHT_DECAY = 1e-5 

# Beta VAE / Capacity Control (Point 3.2 & 3.3)
BETA_START = 0.001 
BETA_END = 1.0 
BETA_ANNEAL_EPOCHS = 100 
C_MAX = 25.0 
C_ANNEAL_EPOCHS_CAPACITY = int(BETA_ANNEAL_EPOCHS * 0.8) # Renamed for clarity

# Classifier Hyperparameters
CLASSIFIER_DENSE_UNITS = [64] 
CLASSIFIER_DROPOUT_RATE = 0.5 
CLASSIFIER_EPOCHS = 150 
CLASSIFIER_LR = 3e-4 
CLASSIFIER_WEIGHT_DECAY = 1e-3 
NUM_BINARY_CLASSES = 2

EARLY_STOPPING_PATIENCE_CVAE = 35 
EARLY_STOPPING_PATIENCE_CLF = 25 

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
        
        for file_path in self.pt_files:
            try:
                content = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False) 
                meta = content["meta"]; data_tensor_all_metrics = content["data"] 
                original_group = meta.get("Group"); mapped_group = self.pathology_mapping.get(original_group)
                if mapped_group is None: continue

                metric_indices = [meta["MetricsOrder"].index(m) for m in self.selected_metrics]
                self.matrices.append(data_tensor_all_metrics[metric_indices, :, :])
                self.ages_raw.append(float(meta.get("Age", np.nan)))
                sex_raw = meta.get("Sex", "Unknown")
                if sex_raw in ("M", "Male"): sex_clean = "Male"
                elif sex_raw in ("F", "Female"): sex_clean = "Female"
                else: sex_clean = "Unknown"
                self.sexes_raw.append(sex_clean)
                self.pathologies_mapped_raw.append(mapped_group)
                self.subject_ids.append(meta.get("SubjectID"))
            except Exception as e: logger.error(f"Skipping file {file_path} due to error: {e}")
        
        if not self.matrices: raise ValueError("No data loaded from provided pt_file_paths.")
        self.matrices = torch.stack(self.matrices).float()
        
        ages_np_temp = np.array(self.ages_raw).astype(float)
        if fit_scalers: 
            self.mean_age = np.nanmean(ages_np_temp) if np.sum(~np.isnan(ages_np_temp)) > 0 else 70.0
        
        mean_age_to_use = self.mean_age if fit_scalers else condition_preprocessors_input['age_mean']
        ages_np = np.nan_to_num(ages_np_temp, nan=mean_age_to_use).reshape(-1,1)
        sexes_np = np.array(self.sexes_raw).reshape(-1, 1)
        pathologies_ohe_np = np.array(self.pathologies_mapped_raw).reshape(-1,1)

        if fit_scalers:
            self.age_scaler = MinMaxScaler(feature_range=(0,1))
            self.sex_encoder = OneHotEncoder(categories=[['Female', 'Male', 'Unknown']], sparse_output=False, handle_unknown='error')
            self.pathology_encoder_ohe_for_cvae = OneHotEncoder(categories=[self.pathology_categories], sparse_output=False, handle_unknown='error')
            
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
        for i in range(self.matrices.shape[1]):
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

    def __len__(self): return len(self.matrices)
    def __getitem__(self, idx):
        binary_label = 1 if self.pathology_labels_str[idx] == "AD" else (0 if self.pathology_labels_str[idx] == "CN" else -1)
        return self.matrices[idx], self.conditions_for_cvae[idx], torch.tensor(binary_label, dtype=torch.long), self.pathology_labels_str[idx]
    
    def get_preprocessors(self): return self.condition_preprocessors, self.data_scalers
    def get_condition_dim(self): return self.conditions_for_cvae.shape[1]

# Dataset Factory Function (Point 2.1)
def make_connectome_dataset(pt_file_paths, selected_metrics, pathology_mapping, 
                            fit_scalers, cond_proc_input=None, data_scalers_input=None):
    logger.info(f"Creating dataset (fit_scalers={fit_scalers}) with {len(pt_file_paths)} files...")
    return ConnectomeDataset(
        pt_file_paths_override=pt_file_paths, 
        selected_metrics=selected_metrics,
        pathology_mapping=pathology_mapping,
        condition_preprocessors_input=cond_proc_input,
        data_scalers_input=data_scalers_input,
        fit_scalers=fit_scalers
    )

# --- ResNet-style Blocks ---
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += shortcut
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=2):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)
        self.shortcut_conv = nn.Conv2d(in_channels, filters3, kernel_size=1, stride=stride, padding=0, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(filters3)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
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
                 decoder_convtranspose_params, final_decoder_convtranspose_params):
        super(CVAE_ResNetEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder_start_conv_channels = decoder_start_conv_channels
        self.decoder_fc_target_h_w = decoder_fc_target_h_w

        encoder_modules = []
        current_channels = img_channels
        current_size = img_res
        encoder_modules.extend([
            nn.Conv2d(current_channels, initial_conv_filters, initial_kernel_size, initial_stride, initial_padding, bias=False),
            nn.BatchNorm2d(initial_conv_filters), nn.SiLU(inplace=True)]) # Using SiLU (Swish)
        current_channels = initial_conv_filters
        current_size = (current_size + 2 * initial_padding - initial_kernel_size) // initial_stride + 1
        if use_max_pool:
            encoder_modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            current_size = (current_size + 2 * 1 - 3) // 2 + 1
        
        for i, (num_blocks, filters_list, stride_for_first) in enumerate(resnet_stages_params):
            # Using GroupNorm instead of BatchNorm (Point 3.1 of review for CVAE architecture)
            # For simplicity in this pass, keeping BatchNorm, but GroupNorm is a good alternative.
            encoder_modules.append(ConvBlock(current_channels, filters_list, resnet_kernel_size, stride=stride_for_first))
            current_channels = filters_list[2]
            if stride_for_first > 1: current_size = (current_size -1) // stride_for_first + 1 
            for _ in range(num_blocks - 1):
                encoder_modules.append(IdentityBlock(current_channels, filters_list, resnet_kernel_size))
        
        encoder_modules.append(nn.AdaptiveAvgPool2d((1,1)))
        self.encoder_conv = nn.Sequential(*encoder_modules)
        self.flattened_size_after_resnet = current_channels 
        
        encoder_fc_layers_list = []
        fc_input_size = self.flattened_size_after_resnet + condition_dim
        for units in encoder_dense_units:
            encoder_fc_layers_list.extend([nn.Linear(fc_input_size, units), nn.SiLU(inplace=True)]) # Using SiLU
            fc_input_size = units
        self.encoder_fc = nn.Sequential(*encoder_fc_layers_list)
        self.fc_z_mean = nn.Linear(fc_input_size, latent_dim)
        self.fc_z_log_var = nn.Linear(fc_input_size, latent_dim)

        decoder_fc_layers_list_dec = []
        current_fc_decoder_input = latent_dim + condition_dim
        decoder_fc_output_size = self.decoder_start_conv_channels * self.decoder_fc_target_h_w * self.decoder_fc_target_h_w
        for units in decoder_dense_units: 
            decoder_fc_layers_list_dec.extend([nn.Linear(current_fc_decoder_input, units), nn.SiLU(inplace=True)]) # Using SiLU
            current_fc_decoder_input = units
        decoder_fc_layers_list_dec.extend([nn.Linear(current_fc_decoder_input, decoder_fc_output_size), nn.SiLU(inplace=True)]) # Using SiLU
        self.decoder_fc = nn.Sequential(*decoder_fc_layers_list_dec)
        
        decoder_conv_t_layers_list = []
        current_channels_dec = self.decoder_start_conv_channels
        for i, (out_channels, kernel, stride, padding, out_pad) in enumerate(decoder_convtranspose_params):
            decoder_conv_t_layers_list.extend([
                nn.ConvTranspose2d(current_channels_dec, out_channels, kernel, stride, padding, output_padding=out_pad, bias=False),
                nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True)]) # Using SiLU
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
                nn.Linear(input_feat, hidden_units), nn.BatchNorm1d(hidden_units),
                nn.ReLU(inplace=True), nn.Dropout(dropout_rate)])
            input_feat = hidden_units
        layers_list.append(nn.Linear(input_feat, num_classes))
        self.network = nn.Sequential(*layers_list)
    def forward(self, latent_vector): return self.network(latent_vector)

# --- Training Loop for Classifier ---
def train_binary_classifier_epoch_v2(classifier, dataloader, optimizer, scaler, criterion, device):
    classifier.train(); total_loss = 0; correct_predictions = 0; total_samples = 0
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
    auc_score = roc_auc_score(all_labels_list, all_probs_positive_class) if len(np.unique(all_labels_list)) > 1 and total_samples > 0 else 0.5
    return avg_loss, avg_acc, all_labels_list, all_probs_positive_class, auc_score

# --- Main Execution ---
if __name__ == "__main__":
    selected_metrics = ["Correlation_FisherZ", "NMI", "GrangerCausality_Directed_FDR"]
    pathology_mapping = {"AD": "AD", "CN": "CN", "MCI": "Other", "LMCI": "Other", "EMCI": "Other"}

    logger.info("--- Initializing Datasets ---")
    all_pt_files_full_list = sorted(glob.glob(os.path.join(PT_FILES_DIR, "*.pt")))
    if not all_pt_files_full_list: raise ValueError("No .pt files found in PT_FILES_DIR")

    temp_pathologies = []
    for file_path in all_pt_files_full_list:
        try:
            content = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
            meta = content["meta"]
            original_group = meta.get("Group")
            mapped_group = pathology_mapping.get(original_group)
            if mapped_group: temp_pathologies.append(mapped_group)
            else: temp_pathologies.append("Unknown_Group_For_Split") 
        except: temp_pathologies.append("Unknown_Group_For_Split")

    indices = list(range(len(all_pt_files_full_list)))
    try:
        train_val_indices, test_indices = train_test_split(
            indices, test_size=0.20, random_state=SEED, stratify=temp_pathologies)
        temp_train_val_pathologies = [temp_pathologies[i] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.20, random_state=SEED, stratify=temp_train_val_pathologies)
        logger.info("Successfully performed stratified splits.")
    except ValueError as e: 
        logger.warning(f"Stratified split failed ({e}), falling back to random split.")
        np.random.shuffle(indices) # Fallback to random shuffle
        num_total = len(all_pt_files_full_list)
        num_test = int(num_total * 0.20); num_val = int((num_total - num_test) * 0.20)
        num_train = num_total - num_test - num_val
        train_indices, val_indices, test_indices = indices[:num_train], indices[num_train:num_train+num_val], indices[num_train+num_val:]

    train_pt_files = [all_pt_files_full_list[i] for i in train_indices]
    val_pt_files = [all_pt_files_full_list[i] for i in val_indices]
    test_pt_files = [all_pt_files_full_list[i] for i in test_indices]

    train_dataset = make_connectome_dataset(train_pt_files, selected_metrics, pathology_mapping, fit_scalers=True)
    CONDITION_DIM = train_dataset.get_condition_dim() # Correctly get after train_dataset is made
    condition_preprocessors, data_scalers = train_dataset.get_preprocessors()

    val_dataset = make_connectome_dataset(val_pt_files, selected_metrics, pathology_mapping, 
                                          fit_scalers=False, cond_proc_input=condition_preprocessors, data_scalers_input=data_scalers)
    test_dataset_final_holdout = make_connectome_dataset(test_pt_files, selected_metrics, pathology_mapping,
                                                         fit_scalers=False, cond_proc_input=condition_preprocessors, data_scalers_input=data_scalers)
    
    train_loader_cvae = DataLoader(train_dataset, batch_size=CVAE_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader_cvae = DataLoader(val_dataset, batch_size=CVAE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    logger.info(f"CVAE Train set: {len(train_dataset)}, CVAE Validation set: {len(val_dataset)}, Final Test set: {len(test_dataset_final_holdout)}")

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
        final_decoder_convtranspose_params=FINAL_DECODER_CONVTRANSPOSE_PARAMS
    ).to(device=DEVICE, memory_format=torch.channels_last if USE_AMP else torch.contiguous_format)
    
    optimizer_cvae = optim.Adam(cvae_model.parameters(), lr=CVAE_LEARNING_RATE, weight_decay=CVAE_WEIGHT_DECAY)
    scheduler_cvae = CosineAnnealingWarmRestarts(optimizer_cvae, T_0=max(10, CVAE_EPOCHS // 10), T_mult=2, eta_min=1e-7, verbose=False) 
    cvae_early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE_CVAE, verbose=True, 
                                       path=os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_best_model.pth"), mode='min')
    cvae_grad_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    logger.info("\n--- Training CVAE with ResNet Encoder (on AD, CN, Other) ---")
    cvae_train_losses_log, cvae_val_losses_log = [], []
    for epoch in range(1, CVAE_EPOCHS + 1):
        current_beta_kl = BETA_END / (1 + np.exp(-6 * (epoch / BETA_ANNEAL_EPOCHS - 0.5)))
        current_capacity_c = min(C_MAX, C_MAX * (epoch / C_ANNEAL_EPOCHS_CAPACITY) ) if C_ANNEAL_EPOCHS_CAPACITY > 0 else C_MAX
        
        train_loss, train_recon, train_kl = train_cvae_epoch(cvae_model, train_loader_cvae, optimizer_cvae, cvae_grad_scaler, current_beta_kl, current_capacity_c, DEVICE)
        val_loss, val_recon, val_kl = validate_cvae_epoch(cvae_model, val_loader_cvae, current_beta_kl, current_capacity_c, DEVICE) 
        scheduler_cvae.step() 

        cvae_train_losses_log.append(train_loss); cvae_val_losses_log.append(val_loss)
        if epoch % 10 == 0 or epoch == CVAE_EPOCHS or epoch == 1:
            logger.info(f"CVAE Epoch {epoch}/{CVAE_EPOCHS}: Beta_KL: {current_beta_kl:.4f} Capacity_C: {current_capacity_c:.2f} LR: {optimizer_cvae.param_groups[0]['lr']:.1e}\n"
                  f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})\n"
                  f"  Val Loss  : {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        cvae_early_stopper(val_loss, cvae_model)
        if cvae_early_stopper.early_stop: logger.info("CVAE Early stopping triggered."); break
    
    logger.info("Loading best CVAE model weights based on validation loss.")
    cvae_model.load_state_dict(torch.load(os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_best_model.pth")))
    plt.figure(figsize=(10,5)); plt.plot(cvae_train_losses_log, label="Train CVAE Loss"); plt.plot(cvae_val_losses_log, label="Val CVAE Loss")
    if cvae_val_losses_log: plt.axvline(np.argmin(cvae_val_losses_log), color='r', linestyle='--', label=f'Best Val Epoch: {np.argmin(cvae_val_losses_log)+1}')
    plt.title("CVAE ResNet Training & Validation Loss"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_loss_plot.png")); plt.close()

    logger.info("\n--- Preparing Data for AD vs CN Classifier ---")
    cvae_model.eval() 
    def extract_latent_and_filter(dataset_subset):
        latent_vectors_list, binary_labels_list = [], []
        for i in range(len(dataset_subset)): 
            matrix, condition, binary_label, pathology_str = dataset_subset[i] 
            if pathology_str in ["AD", "CN"]:
                matrix, condition = matrix.unsqueeze(0).to(DEVICE), condition.unsqueeze(0).to(DEVICE)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    with torch.no_grad(): z_mean, _ = cvae_model.encode(matrix, condition)
                latent_vectors_list.append(z_mean.squeeze(0).cpu())
                binary_labels_list.append(binary_label.cpu()) 
        if not latent_vectors_list: return None, None
        return torch.stack(latent_vectors_list), torch.tensor(binary_labels_list, dtype=torch.long)

    train_latent_clf, train_labels_clf = extract_latent_and_filter(train_dataset)
    val_latent_clf, val_labels_clf = extract_latent_and_filter(val_dataset) 
    test_latent_clf, test_labels_clf = extract_latent_and_filter(test_dataset_final_holdout)

    if train_latent_clf is None or val_latent_clf is None or test_latent_clf is None:
        logger.error("Insufficient AD/CN data in one or more splits for classifier. Exiting."); exit()

    logger.info(f"AD/CN Classifier: Train size={len(train_latent_clf)}, Val size={len(val_latent_clf)}, Test size={len(test_latent_clf)}")
    logger.info(f"Train AD/CN dist: CN={(train_labels_clf==0).sum().item()}, AD={(train_labels_clf==1).sum().item()}")
    logger.info(f"Val AD/CN dist: CN={(val_labels_clf==0).sum().item()}, AD={(val_labels_clf==1).sum().item()}")
    logger.info(f"Test AD/CN dist: CN={(test_labels_clf==0).sum().item()}, AD={(test_labels_clf==1).sum().item()}")

    train_dataset_clf_final = TensorDataset(train_latent_clf, train_labels_clf)
    val_dataset_clf_final = TensorDataset(val_latent_clf, val_labels_clf) 
    test_dataset_clf_final_eval = TensorDataset(test_latent_clf, test_labels_clf) 

    train_loader_clf = DataLoader(train_dataset_clf_final, batch_size=CVAE_BATCH_SIZE, shuffle=True)
    val_loader_clf = DataLoader(val_dataset_clf_final, batch_size=CVAE_BATCH_SIZE, shuffle=False)
    test_loader_clf_final = DataLoader(test_dataset_clf_final_eval, batch_size=CVAE_BATCH_SIZE, shuffle=False) 
    
    binary_classifier = LatentSpaceBinaryClassifier(
        latent_dim=LATENT_DIM, num_classes=NUM_BINARY_CLASSES, hidden_units_list=CLASSIFIER_DENSE_UNITS, dropout_rate=CLASSIFIER_DROPOUT_RATE
    ).to(DEVICE)
    
    # Class weights (Point 1 of review)
    if len(train_labels_clf) > 0 : 
        counts = Counter(train_labels_clf.tolist())
        cn_count = counts.get(0, 0)
        ad_count = counts.get(1, 0)
        if cn_count > 0 and ad_count > 0:
            weight_cn = (cn_count + ad_count) / (2.0 * cn_count) if cn_count > 0 else 1.0
            weight_ad = (cn_count + ad_count) / (2.0 * ad_count) if ad_count > 0 else 1.0
            class_weights_clf = torch.tensor([weight_cn, weight_ad], dtype=torch.float).to(DEVICE)
            criterion_binary_clf = nn.CrossEntropyLoss(weight=class_weights_clf)
            logger.info(f"Using class weights for classifier: CN={weight_cn:.2f}, AD={weight_ad:.2f}")
        else:
            criterion_binary_clf = nn.CrossEntropyLoss()
            logger.info("Using unweighted CrossEntropyLoss for classifier (one class missing in train_labels_clf or empty).")
    else:
        criterion_binary_clf = nn.CrossEntropyLoss()
        logger.info("Using unweighted CrossEntropyLoss for classifier (train_labels_clf is empty).")

    optimizer_binary_clf = optim.Adam(binary_classifier.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WEIGHT_DECAY)
    scheduler_clf = CosineAnnealingWarmRestarts(optimizer_binary_clf, T_0=15, T_mult=2, eta_min=1e-7, verbose=False) 
    classifier_early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE_CLF, verbose=True, 
                                             path=os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_best.pth"), 
                                             mode='max') 
    clf_grad_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    logger.info("\n--- Training AD vs CN Classifier on CVAE Latent Space ---")
    clf_train_losses_log, clf_train_accs_log, clf_val_losses_log, clf_val_accs_log, clf_val_auc_log = [], [], [], [], []
    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        train_loss_clf, train_acc_clf = train_binary_classifier_epoch_v2(binary_classifier, train_loader_clf, optimizer_binary_clf, clf_grad_scaler, criterion_binary_clf, DEVICE)
        val_loss_clf, val_acc_clf, _, _, val_auc_clf = evaluate_binary_classifier(binary_classifier, val_loader_clf, criterion_binary_clf, DEVICE) 
        scheduler_clf.step() 
        
        clf_train_losses_log.append(train_loss_clf); clf_train_accs_log.append(train_acc_clf)
        clf_val_losses_log.append(val_loss_clf); clf_val_accs_log.append(val_acc_clf); clf_val_auc_log.append(val_auc_clf)
        if epoch % 10 == 0 or epoch == CLASSIFIER_EPOCHS or epoch == 1:
            logger.info(f"Binary Classifier Epoch {epoch}/{CLASSIFIER_EPOCHS}: LR: {optimizer_binary_clf.param_groups[0]['lr']:.1e}\n"
                  f"  Train Loss: {train_loss_clf:.4f}, Train Acc: {train_acc_clf:.4f}\n"
                  f"  Val Loss  : {val_loss_clf:.4f}, Val Acc  : {val_acc_clf:.4f}, Val AUC: {val_auc_clf:.4f}")
        
        classifier_early_stopper(val_auc_clf, binary_classifier) # Early stopping on validation AUC
        if classifier_early_stopper.early_stop: logger.info("Classifier Early stopping triggered."); break
    
    logger.info("Loading best binary classifier model weights based on validation metric (AUC).")
    binary_classifier.load_state_dict(torch.load(os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_best.pth")))
    
    plt.figure(figsize=(18, 5)); 
    plt.subplot(1, 3, 1); plt.plot(clf_train_accs_log, label='Train Acc (AD/CN)'); plt.plot(clf_val_accs_log, label='Val Acc (AD/CN)')
    if clf_val_accs_log: plt.axvline(np.argmax(clf_val_accs_log) if clf_val_accs_log else 0, color='r', linestyle='--', label=f'Best Val Epoch (Acc): {np.argmax(clf_val_accs_log)+1 if clf_val_accs_log else 0}')
    plt.title('Classifier Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 3, 2); plt.plot(clf_train_losses_log, label='Train Loss (AD/CN)'); plt.plot(clf_val_losses_log, label='Val Loss (AD/CN)')
    if clf_val_losses_log: plt.axvline(np.argmin(clf_val_losses_log), color='r', linestyle='--', label=f'Best Val Epoch (Loss): {np.argmin(clf_val_losses_log)+1 if clf_val_losses_log else 0}')
    plt.title('Classifier Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 3); plt.plot(clf_val_auc_log, label='Val AUC (AD/CN)')
    if clf_val_auc_log: plt.axvline(np.argmax(clf_val_auc_log), color='r', linestyle='--', label=f'Best Val Epoch (AUC): {np.argmax(clf_val_auc_log)+1 if clf_val_auc_log else 0}')
    plt.title('Classifier Validation AUC'); plt.xlabel('Epochs'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
    
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_training_plot.png")); plt.close()

    logger.info("\n--- Final Evaluation on Test Set (AD vs CN Classifier) ---")
    test_loss_final, test_acc_final, true_test_labels, pred_test_probs_ad, test_auc_final = evaluate_binary_classifier(
        binary_classifier, test_loader_clf_final, criterion_binary_clf, DEVICE) 
    
    logger.info("\n--- Tuning Decision Threshold on Validation Set ---")
    _, _, val_true_labels_for_thresh, val_probs_ad_for_thresh, _ = evaluate_binary_classifier(
        binary_classifier, val_loader_clf, criterion_binary_clf, DEVICE)
    
    best_threshold = 0.5 
    if len(np.unique(val_true_labels_for_thresh)) > 1 and len(val_probs_ad_for_thresh) > 0 : 
        fpr, tpr, thresholds = roc_curve(val_true_labels_for_thresh, val_probs_ad_for_thresh)
        if len(thresholds) > 0 : # Ensure thresholds array is not empty
            youden_j = tpr - fpr
            if len(youden_j) > 0 : 
                best_idx = np.argmax(youden_j)
                best_threshold = thresholds[best_idx]
                logger.info(f"Best decision threshold based on Youden's J on validation set: {best_threshold:.4f}")
            else: logger.warning("Could not determine Youden's J (tpr-fpr array empty). Using 0.5.")
        else: logger.warning("Could not determine Youden's J (thresholds array empty). Using 0.5.")
    else: logger.warning("Not enough data or only one class present in validation set for threshold tuning. Using 0.5.")

    pred_test_labels_for_report = (np.array(pred_test_probs_ad) >= best_threshold).astype(int)

    logger.info(f"Final Test Set Performance with default 0.5 threshold: Accuracy = {test_acc_final:.4f}, AUC = {test_auc_final:.4f}")
    logger.info(f"Final Test Set Accuracy with tuned threshold ({best_threshold:.4f}): {(np.array(true_test_labels) == pred_test_labels_for_report).mean():.4f}")
    logger.info("\nFinal Test Set Classification Report (AD vs CN) with tuned threshold:")
    logger.info(f"\n{classification_report(true_test_labels, pred_test_labels_for_report, target_names=['CN (0)', 'AD (1)'], zero_division=0)}")
    logger.info("\nFinal Test Set Confusion Matrix (AD vs CN) with tuned threshold:")
    logger.info(f"\n{confusion_matrix(true_test_labels, pred_test_labels_for_report)}")

    logger.info("\n--- Sanity Check: Logistic Regression on Latent Features ---")
    from sklearn.linear_model import LogisticRegression
    try:
        if len(train_latent_clf) > 0 and len(np.unique(train_labels_clf.cpu().numpy())) > 1:
            logreg = LogisticRegression(max_iter=4000, class_weight='balanced', random_state=SEED, solver='liblinear')
            logreg.fit(train_latent_clf.cpu().numpy(), train_labels_clf.cpu().numpy())
            
            if len(val_latent_clf) > 0 and len(np.unique(val_labels_clf.cpu().numpy())) > 1:
                val_probs_logreg = logreg.predict_proba(val_latent_clf.cpu().numpy())[:, 1]
                val_auc_logreg = roc_auc_score(val_labels_clf.cpu().numpy(), val_probs_logreg)
                logger.info(f"Logistic Regression - Validation AUC: {val_auc_logreg:.4f}")
            else: logger.warning("Skipping Logistic Regression validation AUC: Insufficient samples or only one class in validation set.")

            if len(test_latent_clf) > 0 and len(np.unique(test_labels_clf.cpu().numpy())) > 1:
                test_probs_logreg = logreg.predict_proba(test_latent_clf.cpu().numpy())[:, 1]
                test_auc_logreg = roc_auc_score(test_labels_clf.cpu().numpy(), test_probs_logreg)
                test_preds_logreg = (test_probs_logreg >= 0.5).astype(int)
                test_acc_logreg = (test_preds_logreg == test_labels_clf.cpu().numpy()).mean()
                logger.info(f"Logistic Regression - Test Accuracy: {test_acc_logreg:.4f}, Test AUC: {test_auc_logreg:.4f}")
                logger.info(f"Logistic Regression - Test Classification Report:\n{classification_report(test_labels_clf.cpu().numpy(), test_preds_logreg, target_names=['CN (0)', 'AD (1)'], zero_division=0)}")
            else: logger.warning("Skipping Logistic Regression test evaluation: Insufficient samples or only one class in test set.")
        else: logger.warning("Skipping Logistic Regression sanity check: Insufficient samples or only one class in training set.")
    except Exception as e: logger.error(f"Error during Logistic Regression sanity check: {e}")

    logger.info("--- Script Finished ---")