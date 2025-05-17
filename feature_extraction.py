#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_extraction.py (v10.1 - Adapted for 116 ROIs and variable timepoints)
--------------------------------------------------------------------------
• Assumes prior detrending; internal detrending is skipped by default.
• Expects a specific number of ROIs in input files (N_ROIs = 116 by default).
• If input files contain more than N_ROIs, use --exclude_roi_indices to select the final set.
  By default, assumes input files already contain exactly 116 ROIs (empty exclusion list).
• Handles variable N_timepoints (e.g., 140, 197, 200).
• Within the final selected set, any ROI with bad signal for a
  subject (initially NaN/zero, or zero-variance post-processing)
  will have its signal zeroed out to ensure numerical stability and
  maintain consistent matrix dimensions. Logging for this is minimized.
• FD-based scrubbing (optional).
• Granger Causality is kept DIRECTED after FDR correction.
• No Partial Correlation.
• Metrics: Correlation, NMI, Directed Granger, DCV.
"""
from __future__ import annotations

import argparse
import logging
import os
import gc
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import torch
import scipy.io as sio
from scipy.signal import butter, filtfilt, detrend as _linear_detrend
from scipy.stats import zscore
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import fdrcorrection
from sklearn.decomposition import PCA

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not found. Using os.cpu_count().")

# =========================================================
#         PREPROCESSING AND METRIC FUNCTIONS
# =========================================================
def poly_detrend(x: np.ndarray, order: int, axis: int = -1) -> np.ndarray:
    if order < 0: return x
    if order == 0: return x - np.mean(x, axis=axis, keepdims=True)
    if order == 1: return _linear_detrend(x, type='linear', axis=axis)
    t = np.arange(x.shape[axis], dtype=np.float64)
    x_T = np.moveaxis(x, axis, -1); coefs = np.polynomial.polynomial.polyfit(t, x_T, order)
    return np.moveaxis(x_T - np.polynomial.polynomial.polyval(t, coefs), -1, axis)

def apply_scrubbing(
    signals: np.ndarray, fd_values: np.ndarray, fd_threshold: float, subject_id: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    if fd_values.ndim > 1 and fd_values.shape[1] > 1: fd_values = fd_values[:,0]
    fd_values = fd_values.flatten()
    if signals.shape[0] != fd_values.shape[0]:
        logging.error(f"[{subject_id}] Sig/FD len mismatch ({signals.shape[0]} vs {fd_values.shape[0]}). No scrubbing.")
        return signals, np.array([],dtype=int), 0
    scrub_mask = fd_values > fd_threshold
    removed_tr_indices = np.where(scrub_mask)[0]; num_trs_scrubbed = len(removed_tr_indices)
    scrubbed_signals = np.delete(signals, removed_tr_indices, axis=0) if num_trs_scrubbed > 0 else signals
    if num_trs_scrubbed > 0: logging.info(f"[{subject_id}] Scrubbed {num_trs_scrubbed} TRs (FD > {fd_threshold}). Orig TRs: {signals.shape[0]} -> New TRs: {scrubbed_signals.shape[0]}")
    if scrubbed_signals.shape[0] == 0 and num_trs_scrubbed > 0: logging.warning(f"[{subject_id}] All TRs scrubbed.")
    return scrubbed_signals, removed_tr_indices, num_trs_scrubbed

def apply_compcor(signals: np.ndarray, n_components: int = 5) -> np.ndarray:
    if signals.shape[1] == 0 or n_components == 0 or signals.shape[0] < n_components :
        if signals.shape[0] < n_components and signals.shape[1] > 0 : logging.warning(f"CompCor: T ({signals.shape[0]}) < n_comp ({n_components}). Skipping.")
        return signals
    num_rois_avail = signals.shape[1]; actual_n_comp = min(n_components, num_rois_avail)
    if actual_n_comp != n_components: logging.debug(f"[{Path(__file__).stem}] CompCor: ROIs ({num_rois_avail}) < n_comp. Using {actual_n_comp}.") # Reduced verbosity
    if actual_n_comp == 0: return signals
    try:
        centered_signals = signals - np.mean(signals, axis=0, keepdims=True)
        varying_mask = np.std(centered_signals, axis=0) > 1e-9
        if not np.any(varying_mask): logging.debug(f"[{Path(__file__).stem}] CompCor: All signals constant. Skipping."); return signals # Reduced verbosity
        num_varying_rois = np.sum(varying_mask)
        if num_varying_rois < actual_n_comp: actual_n_comp = num_varying_rois; logging.debug(f"[{Path(__file__).stem}] CompCor: Varying ROIs ({num_varying_rois}) < n_comp. Using {actual_n_comp}.") # Reduced verbosity
        if actual_n_comp == 0: return signals
        pca = PCA(n_components=actual_n_comp, svd_solver='full'); noise_comps = pca.fit_transform(centered_signals[:, varying_mask])
        cleaned_sigs = signals.copy()
        for i in range(num_rois_avail):
            if not varying_mask[i] and np.std(signals[:,i]) < 1e-9 : continue
            X_reg = np.hstack([noise_comps, np.ones((signals.shape[0], 1))]); beta, _, _, _ = np.linalg.lstsq(X_reg, signals[:,i], rcond=None)
            pred_noise_effect = X_reg[:, :-1] @ beta[:-1] if actual_n_comp > 0 else 0
            cleaned_sigs[:,i] = signals[:,i] - pred_noise_effect
        logging.info(f"Applied CompCor, regressed {actual_n_comp} components.")
        return cleaned_sigs
    except Exception as e: logging.error(f"CompCor error: {e}. No CompCor."); return signals

def preprocess_signals(
    raw_signals: np.ndarray, tr: float, detrend_poly_order: int,
    band: tuple[float, float], apply_zscore_after: bool, subject_id: str
) -> np.ndarray: # Simplified: returns only processed_signals
    """Preprocesses ROI signals. Assumes raw_signals has fixed dimensions. Problematic ROIs are zeroed."""
    if raw_signals.ndim == 1: raw_signals = raw_signals[:, np.newaxis]
    if raw_signals.shape[0] < 2 or raw_signals.shape[1] == 0: return raw_signals

    N_rois = raw_signals.shape[1]; processed_sigs = np.zeros_like(raw_signals, dtype=np.float32)
    samp_freq, nyq_freq = 1.0/tr, (1.0/tr)/2.0
    try: b, a = butter(4, [b_val/nyq_freq for b_val in band], btype='band')
    except ValueError as e_butt: logging.error(f"[{subject_id}] Butter design fail (Nyq={nyq_freq},Band={band}): {e_butt}. All sigs for subj zeroed."); return processed_sigs

    for i in range(N_rois):
        roi_sig = raw_signals[:,i].copy(); proc_roi_sig = roi_sig # Start with original (could be all zeros)
        if np.std(roi_sig) > 1e-9: # Only process if there's variance
            sig_detrended = poly_detrend(roi_sig, order=detrend_poly_order) if detrend_poly_order >= 0 else roi_sig
            if np.std(sig_detrended) > 1e-9:
                try: proc_roi_sig = filtfilt(b, a, sig_detrended)
                except ValueError as e_filt: logging.debug(f"[{subject_id}] ROI {i} Filter fail: {e_filt}. Using detrended."); proc_roi_sig = sig_detrended # Reduced verbosity
            else: proc_roi_sig = sig_detrended # All zero after detrend
        processed_sigs[:,i] = proc_roi_sig

    if apply_zscore_after and processed_sigs.shape[0] > 0:
        try:
            for i in range(N_rois): # Z-score column-wise
                if np.std(processed_sigs[:,i]) > 1e-9: processed_sigs[:,i] = zscore(processed_sigs[:,i])
                else: processed_sigs[:,i] = 0.0 # Ensure constant (esp. if not already 0) becomes 0
        except Exception as e_z: logging.warning(f"[{subject_id}] Z-score issue: {e_z}. Some ROIs might not be z-scored/zeroed correctly.")

    # Final check: ensure any ROI that ended up with no variance is explicitly all zeros
    for i in range(N_rois):
        if np.std(processed_sigs[:,i]) < 1e-9: processed_sigs[:,i] = 0.0

    return processed_sigs

def compute_correlation_matrix(signals: np.ndarray) -> np.ndarray: # Simplified, assumes input `signals` is fine
    N_rois = signals.shape[1]; corr_matrix = np.zeros((N_rois, N_rois), dtype=np.float32)
    if signals.shape[0] < 2: return corr_matrix
    valid_mask = np.std(signals, axis=0) > 1e-9; num_valid = np.sum(valid_mask)
    if num_valid < 2: return corr_matrix
    valid_sigs = signals[:, valid_mask]; z_sigs = np.nan_to_num(zscore(valid_sigs))
    corr_sub = np.clip(np.nan_to_num(np.corrcoef(z_sigs, rowvar=False)), -0.999999, 0.999999)
    fisher_z_sub = 0.5 * np.log((1 + corr_sub) / (1 - corr_sub))
    valid_idx = np.where(valid_mask)[0]; r_idx_f, c_idx_f = np.meshgrid(valid_idx, valid_idx); r_idx_s, c_idx_s = np.meshgrid(np.arange(num_valid),np.arange(num_valid))
    corr_matrix[r_idx_f, c_idx_f] = fisher_z_sub[r_idx_s, c_idx_s]
    return corr_matrix

def compute_nmi_matrix(signals: np.ndarray, num_bins_override: Optional[int] = None) -> np.ndarray: # Simplified
    T, N = signals.shape; nmi_matrix = np.zeros((N, N), dtype=np.float32)
    if T < 2: return nmi_matrix
    valid_mask = np.std(signals, axis=0) > 1e-9; num_valid = np.sum(valid_mask)
    if num_valid < 1: return nmi_matrix
    valid_sigs = signals[:, valid_mask]; N_val = valid_sigs.shape[1]
    num_bins = num_bins_override if num_bins_override is not None else int(np.clip(np.sqrt(T), 8, 16))
    if num_bins <=0 : num_bins = 2
    edges = np.percentile(valid_sigs, np.linspace(0, 100, num_bins + 1), axis=0).T
    disc_sigs = np.array([np.clip(np.digitize(valid_sigs[:,i], np.unique(edges[i])[1:-1]),0,num_bins-1) if len(np.unique(edges[i]))>1 else np.zeros(T,dtype=np.int16) for i in range(N_val)]).T
    marg_H = np.zeros(N_val,dtype=np.float32)
    for i in range(N_val):
        counts = np.bincount(disc_sigs[:,i],minlength=num_bins); probs = counts[counts>0]/T
        bias_m = (len(probs)-1)/(2*T) if T>0 else 0.0; marg_H[i] = max(0.0, -np.sum(probs*np.log(probs+1e-12)) - bias_m)
    mi_sub = np.zeros((N_val,N_val),dtype=np.float32)
    for i in range(N_val):
        mi_sub[i,i] = marg_H[i]
        for j in range(i+1,N_val):
            jh = np.histogram2d(disc_sigs[:,i],disc_sigs[:,j],bins=num_bins,range=[[-0.5,num_bins-0.5]]*2)[0]
            jp = jh[jh>0]/T; bias_j = (len(jp)-1)/(2*T) if T>0 else 0.0
            joint_H_corr = max(0.0, -np.sum(jp*np.log(jp+1e-12)) - bias_j)
            mi_sub[i,j]=mi_sub[j,i]=max(0,marg_H[i]+marg_H[j]-joint_H_corr)
    nmi_sub = np.clip(mi_sub/(np.sqrt(np.outer(np.maximum(marg_H,1e-12),np.maximum(marg_H,1e-12)))+1e-12),0.0,1.0); np.fill_diagonal(nmi_sub,1.0)
    valid_idx = np.where(valid_mask)[0]; r_idx_f, c_idx_f = np.meshgrid(valid_idx,valid_idx); r_idx_s,c_idx_s = np.meshgrid(np.arange(N_val),np.arange(N_val))
    nmi_matrix[r_idx_f,c_idx_f] = nmi_sub[r_idx_s,c_idx_s]
    if num_valid == 1 and valid_idx.size > 0 : nmi_matrix[valid_idx[0],valid_idx[0]] = 1.0
    return nmi_matrix

def _granger_worker(sigs_slice: np.ndarray, t_local: int, s_local: int, lag: int) -> Tuple[int,int,float]:
    try:
        data = sigs_slice[:, [t_local, s_local]]
        if len(data) < 2*lag + 10:
            return t_local, s_local, 0.0
        pval = grangercausalitytests(data, [lag], verbose=False)[lag][0]['ssr_ftest'][1]
        return t_local, s_local, -np.log(pval if pval > np.finfo(float).tiny else np.finfo(float).tiny)
    except:
        return t_local, s_local, 0.0


def compute_gc_matrix(signals: np.ndarray, max_lag: int, gc_workers: int) -> np.ndarray: # Simplified
    T,N = signals.shape; gc_full = np.zeros((N,N),dtype=np.float32)
    if T==0: return gc_full
    valid_mask=np.std(signals,axis=0)>1e-9; N_val=np.sum(valid_mask)
    if N_val < 2: return gc_full
    valid_sigs = signals[:,valid_mask]; T_val = valid_sigs.shape[0]
    if T_val < max_lag*2+10: logging.debug(f"GC: T_valid ({T_val}) short for lag ({max_lag}), N_valid={N_val}."); return gc_full # Reduced verbosity
    gc_sub=np.zeros((N_val,N_val),dtype=np.float32); args=[(valid_sigs,t,s,max_lag) for t in range(N_val) for s in range(N_val) if t!=s]
    if not args: return gc_full
    with ProcessPoolExecutor(max_workers=gc_workers) as pool:
        futs=[pool.submit(_granger_worker,*a) for a in args]
        for f in as_completed(futs):
            try: t,s,val=f.result(); gc_sub[t,s]=val
            except: pass
    valid_idx=np.where(valid_mask)[0]; r_idx_f,c_idx_f=np.meshgrid(valid_idx,valid_idx); r_idx_s,c_idx_s=np.meshgrid(np.arange(N_val),np.arange(N_val))
    gc_full[r_idx_f,c_idx_f]=gc_sub[r_idx_s,c_idx_s]
    return gc_full

def threshold_gc_fdr_directed(gc_neg_log_p: np.ndarray, alpha: float=0.05) -> np.ndarray:
    N=gc_neg_log_p.shape[0]; fdr_gc=np.zeros_like(gc_neg_log_p)
    if N<2: return fdr_gc
    p_vals,indices=[],[]
    for r in range(N):
        for c in range(N):
            if r==c or (gc_neg_log_p[r,c]==0 and not (np.isinf(gc_neg_log_p[r,c]) and gc_neg_log_p[r,c]>0)): continue
            val=gc_neg_log_p[r,c]; p=1.0 if val==0 and not (np.isinf(val)and val>0) else (np.finfo(float).tiny if np.isinf(val)and val>0 else np.exp(-val))
            p_vals.append(p); indices.append((r,c))
    if not p_vals: logging.debug("DIRECTED GC: No connections to FDR correct."); return fdr_gc # Reduced verbosity
    reject,_=fdrcorrection(np.array(p_vals),alpha=alpha,method='indep')
    sig_n=0;
    for i,(r,c) in enumerate(indices):
        if reject[i]: fdr_gc[r,c]=gc_neg_log_p[r,c]; sig_n+=1
    logging.info(f"FDR (alpha={alpha}) DIRECTED GC: {sig_n}/{len(p_vals)} connections significant.")
    return fdr_gc

def compute_dcv_matrix(signals: np.ndarray, win_s: float, step_s: float, tr: float) -> np.ndarray: # Simplified
    T,N=signals.shape; dcv=np.zeros((N,N),dtype=np.float32)
    if T==0: return dcv
    win_pts,step_pts=int(win_s/(tr if tr>0 else 1)),max(1,int(step_s/(tr if tr>0 else 1)))
    if win_pts<2 or T<win_pts: logging.debug(f"DCV: Invalid win/TRs(T={T},WinPts={win_pts})."); return dcv # Reduced verbosity
    dyn_corrs=[compute_correlation_matrix(signals[s:s+win_pts,:]) for s in range(0,T-win_pts+1,step_pts) if signals[s:s+win_pts,:].shape[0]>=2]
    if not dyn_corrs: logging.debug("No valid segments for DCV."); return dcv # Reduced verbosity
    stack=np.stack(dyn_corrs,axis=2); std_dev,mean_abs=np.std(stack,axis=2),np.abs(np.mean(stack,axis=2))
    mask=mean_abs>1e-12; dcv[mask]=std_dev[mask]/mean_abs[mask]; return np.nan_to_num(dcv)

# =========================================================
#                 INDIVIDUAL SUBJECT PROCESSING
# =========================================================
def run_one_subject(args_tuple: Tuple[str, dict, dict, dict, dict, dict]) -> Tuple[str, bool, Optional[str], Optional[int], Optional[int]]:
    subject_id, cfg, research_lk, age_lk, sex_lk, mmse_lk = args_tuple
    logging.info(f"[{subject_id}] Starting.")
    mat_path = Path(cfg['roi_signals_dir']) / cfg['roi_filename_template'].format(subject_id=subject_id)

    input_rois_in_mat = cfg.get('input_num_rois', 116) # MODIFIED DEFAULT
    excluded_1based_indices = cfg.get('exclude_roi_indices', []) # MODIFIED DEFAULT
    num_rois_final_expected = input_rois_in_mat - len(excluded_1based_indices)

    num_problematic_rois_in_selected_set = 0 # Simplified tracking

    try:
        data = sio.loadmat(mat_path)
        loaded_sigs = data.get("ROISignals", data.get("signals", data.get("roi_signals", data.get("ROIsignals"))))
        if loaded_sigs is None: return subject_id, False, f"Signal key missing: {mat_path}", num_rois_final_expected, -1
        if not isinstance(loaded_sigs, np.ndarray): return subject_id, False, f"Sigs not ndarray: {mat_path}", num_rois_final_expected, -1
        if loaded_sigs.ndim == 1: loaded_sigs = loaded_sigs[:,np.newaxis]
        if loaded_sigs.shape[0] < loaded_sigs.shape[1] and loaded_sigs.shape[0] > 1: loaded_sigs = loaded_sigs.T

        N_loaded = loaded_sigs.shape[1]
        if N_loaded != input_rois_in_mat:
            logging.error(f"[{subject_id}] MAT file has {N_loaded} ROIs, expected {input_rois_in_mat} based on --input_num_rois. Skipping. File: {mat_path}")
            return subject_id, False, f"MAT ROI count error: got {N_loaded}, expected {input_rois_in_mat}", num_rois_final_expected, -1

        selection_mask = np.ones(N_loaded, dtype=bool)
        if excluded_1based_indices:
            exclude_0based = [idx - 1 for idx in excluded_1based_indices if 0 < idx <= N_loaded]
            selection_mask[exclude_0based] = False

        raw_sigs_selected = loaded_sigs[:, selection_mask]
        N_after_selection = raw_sigs_selected.shape[1]

        if N_after_selection != num_rois_final_expected: 
            logging.critical(f"[{subject_id}] Internal logic error or misconfiguration: After excluding ROIs, got {N_after_selection}, but calculated final should be {num_rois_final_expected} (from input_num_rois - len(exclude_roi_indices)). Skipping.")
            return subject_id, False, "ROI selection count mismatch (internal logic error or misconfiguration)", num_rois_final_expected, -1
        logging.info(f"[{subject_id}] Loaded {N_loaded} ROIs, selected {N_after_selection} by excluding 1-based: {excluded_1based_indices if excluded_1based_indices else 'None'}.")

    except Exception as e: return subject_id, False, f"MAT load/ROI selection error {mat_path}: {e}", num_rois_final_expected, -1

    orig_trs = raw_sigs_selected.shape[0]
    #logging.info(f"[{subject_id}] Initial timepoints: {orig_trs}. Expected N_ROIs: {num_rois_final_expected}") # Example: N_timepoints can be 140, 197, or 200

    sigs_for_proc = raw_sigs_selected.copy()

    # Zero out initial problematic ROIs *within the selected set*
    init_problem_mask = np.isnan(sigs_for_proc).all(axis=0) | (np.abs(sigs_for_proc) < 1e-9).all(axis=0)
    if np.any(init_problem_mask):
        sigs_for_proc[:, init_problem_mask] = 0.0
        num_problematic_rois_in_selected_set += np.sum(init_problem_mask)
        logging.debug(f"[{subject_id}] {np.sum(init_problem_mask)} initial problem ROIs in selected set zeroed.") # Reduced verbosity
    if sigs_for_proc.shape[0] == 0: return subject_id, False, "No timepoints.", num_rois_final_expected, 0

    sigs_scrubbed, n_trs_scrubbed, scrub_tr_idx = sigs_for_proc, 0, np.array([],dtype=int)
    num_trs_post_scrub = orig_trs
    if cfg.get('apply_fd_scrubbing', False): # FD Scrubbing logic
        fd_path_str = cfg.get('fd_filepath_template',""); fd_file_res = None
        if "{subject_id}" in fd_path_str: fd_file_res = Path(fd_path_str.format(subject_id=subject_id))
        elif cfg.get(f"fd_path_{subject_id}"): fd_file_res = Path(cfg[f"fd_path_{subject_id}"])
        if fd_file_res and fd_file_res.exists():
            try:
                fd_vals=None; fd_col=cfg.get('fd_column_name_in_file')
                if fd_file_res.suffix in ['.tsv','.csv'] and fd_col: df=pd.read_csv(fd_file_res,sep='\t' if fd_file_res.suffix=='.tsv' else ','); fd_vals=df[fd_col].values if fd_col in df else None
                elif fd_file_res.suffix=='.txt' or not fd_col: fd_vals=pd.read_csv(fd_file_res,header=None).values
                if fd_vals is not None:
                    sigs_scrubbed, scrub_tr_idx, n_trs_scrubbed = apply_scrubbing(sigs_for_proc,fd_vals,cfg['fd_threshold'],subject_id)
                    if sigs_scrubbed.shape[0]==0: return subject_id,False,"All TRs scrubbed.",num_rois_final_expected,0
                # else: logging.debug(f"[{subject_id}] FD extract fail {fd_file_res}. No scrubbing.") # Reduced verbosity
            except Exception as e: logging.error(f"[{subject_id}] FD error {fd_file_res}: {e}. No scrubbing.")
        elif cfg.get('apply_fd_scrubbing',False): logging.warning(f"[{subject_id}] FD scrub ON, file {str(fd_file_res)} missing.")
    num_trs_post_scrub = sigs_scrubbed.shape[0]
    if num_trs_post_scrub < 15: return subject_id,False,f"TRs ({num_trs_post_scrub}) < 15 post-scrub.", num_rois_final_expected, num_trs_post_scrub

    sigs_for_metrics = sigs_scrubbed
    try:
        detrend_ord = -1 if cfg.get('data_is_pre_detrended',True) else cfg['detrend_poly_order']
        if detrend_ord == -1: logging.info(f"[{subject_id}] Skipping internal detrend (pre-detrended data).")
        sigs_proc = preprocess_signals(sigs_scrubbed,cfg['tr'],detrend_ord,cfg['filter_band'],True,subject_id) # preprocess_signals now returns only signals
        if cfg['apply_compcor']: sigs_proc = apply_compcor(sigs_proc,cfg['compcor_n_components'])
        sigs_for_metrics = sigs_proc.copy()
        # Final check for zero-variance is now inside preprocess_signals and metric functions.
        # Update count of problematic ROIs based on final state
        final_problem_mask = np.std(sigs_for_metrics, axis=0) < 1e-9
        num_problematic_rois_in_selected_set = np.sum(final_problem_mask) # This is total that ended up zero
        if num_problematic_rois_in_selected_set > 0 : logging.debug(f"[{subject_id}] {num_problematic_rois_in_selected_set} ROIs in selected set are zero-variance before metrics.") # Reduced
    except Exception as e: return subject_id,False,f"Main preprocess error: {e}",num_rois_final_expected,num_trs_post_scrub

    T_final, N_dim = sigs_for_metrics.shape
    if N_dim!=num_rois_final_expected: return subject_id,False,f"ROI dim error: {N_dim} vs {num_rois_final_expected}",num_rois_final_expected,T_final
    if T_final < 15: return subject_id,False,f"TRs ({T_final}) < 15 for metrics.",num_rois_final_expected,T_final
    logging.info(f"[{subject_id}] Sigs for metrics: T={T_final}, N_ROIs_dim={N_dim}.") # N_dim should be 116

    metrics, names = [], []
    try:
        metrics.append(compute_correlation_matrix(sigs_for_metrics)); names.append("Correlation_FisherZ")
        metrics.append(compute_nmi_matrix(sigs_for_metrics,cfg['nmi_num_bins'])); names.append("NMI")
        gc_raw = compute_gc_matrix(sigs_for_metrics,cfg['gc_max_lag'],cfg['gc_n_workers_internal'])
        gc_fin = threshold_gc_fdr_directed(gc_raw,cfg['gc_fdr_alpha']) if cfg['apply_gc_fdr'] else gc_raw
        metrics.append(gc_fin); names.append("GrangerCausality_Directed_FDR" if cfg['apply_gc_fdr'] else "GrangerCausality_Directed")
        metrics.append(compute_dcv_matrix(sigs_for_metrics,cfg['dcv_fast_window_seconds'],cfg['dcv_fast_step_seconds'],cfg['tr'])); names.append("DCV_Fast_CV")
        metrics.append(compute_dcv_matrix(sigs_for_metrics,cfg['dcv_slow_window_seconds'],cfg['dcv_slow_step_seconds'],cfg['tr'])); names.append("DCV_Slow_CV")
        tensor = np.stack(metrics,axis=0).astype(np.float32)
        for k in range(tensor.shape[0]): np.fill_diagonal(tensor[k],0.0)
    except Exception as e: return subject_id,False,f"Metric error: {e}",num_rois_final_expected,T_final

    res_grp = research_lk.get(subject_id,"Unk"); out_path = Path(cfg['output_dir'])/f"{res_grp if res_grp in {'AD','CN'} else 'Other'}_tensor_{subject_id}.pt"
    nmi_b = cfg['nmi_num_bins'] if cfg['nmi_num_bins'] is not None else (int(np.clip(np.sqrt(T_final),8,16)) if T_final>0 else 0)
    meta = {
        "SubjectID":subject_id,"Group":res_grp,"Age":age_lk.get(subject_id,np.nan),"Sex":sex_lk.get(subject_id,"Unk"),"MMSE":mmse_lk.get(subject_id,np.nan),
        "NumROIsFinalDim":N_dim, "OriginalInputROIsLoaded":N_loaded if 'N_loaded' in locals() else -1, # Should be 116 if input file is correct
        "ConfigInputROIsParam": input_rois_in_mat, # Value from --input_num_rois
        "ConfigExcludedROIs_1Based": excluded_1based_indices,
        "NumProblematicROIsInSelectedSet": num_problematic_rois_in_selected_set, # Simplified
        "OriginalTimePoints":orig_trs,"TimePointsAfterScrubbing":num_trs_post_scrub,"TimePointsForMetrics":T_final,"NumTRsScrubbed":n_trs_scrubbed,
        "FD_Threshold":cfg.get('fd_threshold') if cfg.get('apply_fd_scrubbing') else None,
        "FD_ScrubbingApplied":cfg.get('apply_fd_scrubbing',False),"RemovedTRs_Indices_Scrubbing":scrub_tr_idx.tolist(),
        "TR_seconds":cfg['tr'],"FilterBand_Hz":cfg['filter_band'],
        "InternalDetrendPolyOrder":"Skipped" if detrend_ord==-1 else detrend_ord,
        "AppliedCompCor":cfg['apply_compcor'],"CompCorNComponents":cfg['compcor_n_components'] if cfg['apply_compcor'] else None,
        "GC_MaxLag":cfg['gc_max_lag'],"GC_FDR_Applied":cfg['apply_gc_fdr'],"GC_FDR_Alpha":cfg['gc_fdr_alpha'] if cfg['apply_gc_fdr'] else None,
        "DCV_FastWindow_s":cfg['dcv_fast_window_seconds'],"DCV_FastStep_s":cfg['dcv_fast_step_seconds'],
        "DCV_SlowWindow_s":cfg['dcv_slow_window_seconds'],"DCV_SlowStep_s":cfg['dcv_slow_step_seconds'],
        "NMI_NumBinsUsed":nmi_b,"MetricsOrder":names
    }
    try: torch.save({"data":torch.tensor(tensor,dtype=torch.float32),"meta":meta},out_path); return subject_id,True,str(out_path),N_dim,T_final
    except Exception as e: return subject_id,False,f"Save error: {e}",N_dim,T_final

# =========================================================
#                         MAIN
# =========================================================
def main(config: argparse.Namespace):
    out_dir = Path(config.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    log_f = out_dir/f"pipeline_log_v10.1_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log" # Updated version
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s [%(processName)s]: %(message)s', handlers=[logging.FileHandler(log_f),logging.StreamHandler()])
    logger = logging.getLogger(__name__); logger.info("Pipeline v10.1 (Adapted for 116 ROIs and variable timepoints)..."); logger.info(f"Config: {vars(config)}")

    try: # Load Subject Info
        subj_df = pd.read_csv(config.subject_list_csv); subj_df["SubjectID"] = subj_df["SubjectID"].astype(str).str.strip()
        sids = subj_df["SubjectID"].unique().tolist()
        lkps = {c: pd.Series(subj_df[c].values,index=subj_df.SubjectID.astype(str)).to_dict() for c in ["ResearchGroup","Age","Sex","MMSE"] if c in subj_df}
        if config.fd_column_in_csv and config.fd_column_in_csv in subj_df.columns: # FD Path Handling
            fd_base = Path(config.fd_files_base_dir if config.fd_files_base_dir else Path("."))
            lkps[config.fd_column_in_csv] = {
                r["SubjectID"]: str(fd_base / Path(r[config.fd_column_in_csv])) if pd.notna(r[config.fd_column_in_csv]) and not Path(r[config.fd_column_in_csv]).is_absolute() else (str(Path(r[config.fd_column_in_csv])) if pd.notna(r[config.fd_column_in_csv]) else None)
                for _, r in subj_df.iterrows()
            }
            logger.info(f"FD paths from CSV col '{config.fd_column_in_csv}'.")
    except Exception as e: logger.error(f"Subj CSV load error {config.subject_list_csv}: {e}"); return

    sids_proc = sids[:] ; qc_skip_n = 0 # QC Skipping
    if config.qc_report_csv:
        try:
            qc_df=pd.read_csv(config.qc_report_csv); qc_df['SubjectID']=qc_df['SubjectID'].astype(str).str.strip()
            if config.skip_flagged_subjects:
                mflags,sflags = qc_df.get('MotionFlag',pd.Series(dtype=bool)), qc_df.get('SignalQualityFlag',pd.Series(dtype=bool))
                flag_set = set(qc_df[mflags==True]['SubjectID'])|set(qc_df[sflags==True]['SubjectID'])
                sids_proc=[s for s in sids_proc if s not in flag_set]; qc_skip_n=len(sids)-len(sids_proc)
                logger.info(f"QC: Skipped {qc_skip_n}. Processing {len(sids_proc)}.")
        except Exception as e: logger.warning(f"QC error {config.qc_report_csv}: {e}. No QC skip.")
    if not sids_proc: logger.info("No subjects."); return

    n_cores = (psutil.cpu_count(logical=False) or psutil.cpu_count() or 4) if PSUTIL_AVAILABLE else (os.cpu_count() or 4)
    workers = max(1,config.subject_workers); gc_int_workers = max(1,n_cores//workers if workers>0 else 1)
    cfg_base = vars(config).copy(); cfg_base['gc_n_workers_internal'] = gc_int_workers
    # Ensure the run_one_subject uses the potentially overridden config for these critical params
    cfg_base['input_num_rois'] = config.input_num_rois
    cfg_base['exclude_roi_indices'] = config.exclude_roi_indices


    all_tasks = [] # Prepare tasks
    for sid_val in sids_proc:
        subj_cfg = cfg_base.copy()
        if config.fd_column_in_csv and lkps.get(config.fd_column_in_csv,{}).get(sid_val): subj_cfg[f"fd_path_{sid_val}"]=lkps[config.fd_column_in_csv][sid_val]
        all_tasks.append((sid_val,subj_cfg,lkps.get("ResearchGroup",{}),lkps.get("Age",{}),lkps.get("Sex",{}),lkps.get("MMSE",{})))

    s_count,fail_list,meta_list = 0,[],[] # Initialize counters and lists
    # This calculation now directly uses the (potentially new default) config values
    calculated_final_rois = config.input_num_rois - len(config.exclude_roi_indices)
    logger.info(f"Expecting input ROIs per file: {config.input_num_rois}, excluding {len(config.exclude_roi_indices)} (from CLI or new defaults). Final ROI dimension expected: {calculated_final_rois}")


    for i_batch in range(0,len(all_tasks),config.subjects_per_batch): # Batch processing loop
        cur_batch = all_tasks[i_batch:i_batch+config.subjects_per_batch]
        b_num = (i_batch//config.subjects_per_batch)+1; logger.info(f"--- Batch {b_num}/{ (len(all_tasks)+config.subjects_per_batch-1)//config.subjects_per_batch } (Size: {len(cur_batch)}) ---")
        with ProcessPoolExecutor(max_workers=workers) as pool: # Process pool for current batch
            futs={pool.submit(run_one_subject,task_arg):task_arg[0] for task_arg in cur_batch}
            for fut_item in tqdm(as_completed(futs), total=len(cur_batch), desc=f"Batch {b_num}", ncols=100, leave=False):
                s_id_done = futs[fut_item]
                try:
                    _, succ, mpath, nroi_meta, ntr_meta = fut_item.result()
                    if succ:
                        s_count += 1
                        try:
                            meta_list.append(torch.load(mpath, weights_only=False)['meta'])
                        except Exception as eL:
                            meta_list.append({
                                "SubjectID": s_id_done,
                                "NumROIsFinalDim": nroi_meta,
                                "TimePointsForMetrics": ntr_meta,
                                "ErrorLoadingMeta": True,
                                "LoadError": str(eL)
                            })
                            logger.error(f"[{s_id_done}] Err load meta {mpath}:{eL}")
                    else:
                        fail_list.append((s_id_done, mpath))
                except Exception as eF:
                    fail_list.append((s_id_done, f"Worker critical:{eF}"))

                except Exception as eF: fail_list.append((s_id_done,f"Worker critical:{eF}"))
        del futs; gc.collect(); logger.info(f"Batch {b_num} done. Mem: {psutil.Process(os.getpid()).memory_info().rss/1024**2:.2f}MB" if PSUTIL_AVAILABLE else "")

    logger.info(f"========== Pipeline Finished: {s_count}/{len(all_tasks)} succeeded ==========") # Summary
    if qc_skip_n: logger.info(f"{qc_skip_n} subjects skipped by QC.")
    for sid,reason in fail_list: logger.warning(f"  FAILED {sid}: {reason}")
    if meta_list: # Post-hoc checks
        rois_final_dim_counts = pd.Series([m.get('NumROIsFinalDim',-1) for m in meta_list]).value_counts()
        trs_final_counts = pd.Series([m.get('TimePointsForMetrics',-1) for m in meta_list]).value_counts()
        logger.info(f"Final ROI Dim counts (should consistently be {calculated_final_rois}):\n{rois_final_dim_counts.to_string()}")
        logger.info(f"Final TRs counts (may vary, e.g., 140, 197, 200, or less if scrubbed):\n{trs_final_counts.to_string()}")
        all_match_calc_rois = all(m.get('NumROIsFinalDim')==calculated_final_rois for m in meta_list if m.get('NumROIsFinalDim',-1)!=-1)
        if not all_match_calc_rois or len(rois_final_dim_counts)>1: logger.warning("NumROIsFinalDim VARIES or MISMATCHES calculated target. Check logs for errors in individual subject processing or .mat file content.")
        if len(trs_final_counts)>1: logger.info("TimePointsForMetrics VARIES. This is expected if input TRs vary or due to subject-specific scrubbing.")
    logger.info(f"Outputs in: {config.output_dir}. Log: {log_f}")

# =========================================================
#                     CLI PARSER
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fMRI ROI Feature Extraction (v10.1 - Adapted for 116 ROIs).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_dir",type=str,default=".",help="Base for relative paths.")
    parser.add_argument("--subject_list_csv",required=True,type=str)
    parser.add_argument("--roi_signals_dir",required=True,type=str)
    parser.add_argument("--roi_filename_template",type=str,default="ROISignals_{subject_id}.mat")
    parser.add_argument("--output_dir",required=True,type=str)
    # MODIFIED DEFAULTS FOR ROI HANDLING
    parser.add_argument("--input_num_rois", type=int, default=116,
                        help="Number of ROIs expected in input .mat files (e.g., 116 for your specific atlas).")
    parser.add_argument("--exclude_roi_indices", type=int, nargs='+', default=[],
                        help="List of 1-based ROI indices to EXCLUDE. If --input_num_rois is the final desired count (e.g., 116), this should be empty.")

    parser.add_argument("--qc_report_csv",type=str,default=None); parser.add_argument("--skip_flagged_subjects",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--apply_fd_scrubbing",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--fd_filepath_template",type=str,default=""); parser.add_argument("--fd_column_name_in_file",type=str,default="framewise_displacement")
    parser.add_argument("--fd_column_in_csv",type=str,default=""); parser.add_argument("--fd_files_base_dir",type=str,default="")
    parser.add_argument("--fd_threshold",type=float,default=0.5)
    parser.add_argument("--tr",type=float,required=True)
    parser.add_argument("--data_is_pre_detrended",action=argparse.BooleanOptionalAction,default=True,help="Set if data ALREADY detrended (skips internal detrend).")
    parser.add_argument("--detrend_poly_order",type=int,default=1,help="Order for internal detrend, ONLY if --data_is_pre_detrended is False.")
    parser.add_argument("--filter_band_low",type=float,default=0.01); parser.add_argument("--filter_band_high",type=float,default=0.08)
    parser.add_argument("--apply_compcor",action=argparse.BooleanOptionalAction,default=False); parser.add_argument("--compcor_n_components",type=int,default=5)
    parser.add_argument("--dcv_fast_window_seconds",type=float,default=30.0); parser.add_argument("--dcv_fast_step_seconds",type=float,default=3.0)
    parser.add_argument("--dcv_slow_window_seconds",type=float,default=90.0); parser.add_argument("--dcv_slow_step_seconds",type=float,default=9.0)
    parser.add_argument("--nmi_num_bins",type=int,default=None)
    parser.add_argument("--gc_max_lag",type=int,default=2); parser.add_argument("--apply_gc_fdr",action=argparse.BooleanOptionalAction,default=True); parser.add_argument("--gc_fdr_alpha",type=float,default=0.05)
    phys_cores_def=(psutil.cpu_count(logical=False) or psutil.cpu_count() or 4) if PSUTIL_AVAILABLE else (os.cpu_count() or 4)
    parser.add_argument("--subject_workers",type=int,default=max(1,phys_cores_def//2)); parser.add_argument("--subjects_per_batch",type=int,default=8)

    cfg=parser.parse_args(); cfg.filter_band=(cfg.filter_band_low,cfg.filter_band_high)
    base_p=Path(cfg.base_dir).resolve()
    for arg_p in ["subject_list_csv","roi_signals_dir","output_dir","qc_report_csv","fd_files_base_dir","fd_filepath_template"]:
        val_p=getattr(cfg,arg_p)
        if val_p and str(val_p).strip(): path_obj=Path(val_p)
        if val_p and str(val_p).strip() and not path_obj.is_absolute() and not (arg_p=="fd_filepath_template" and "{subject_id}" in str(path_obj) and not cfg.fd_column_in_csv): setattr(cfg,arg_p,str(base_p/path_obj))
        elif val_p and str(val_p).strip() and not path_obj.is_absolute() and arg_p=="fd_filepath_template" and "{subject_id}" in str(path_obj) : setattr(cfg,arg_p,str(base_p/path_obj))
        elif arg_p in ["qc_report_csv","fd_files_base_dir"]: setattr(cfg,arg_p,None if not val_p or str(val_p).strip()=='' else str(Path(val_p).resolve()))
    if cfg.data_is_pre_detrended and cfg.detrend_poly_order!=-1 and cfg.detrend_poly_order>=0: print(f"INFO: --data_is_pre_detrended. Internal detrend skipped (ignoring --detrend_poly_order={cfg.detrend_poly_order}).")
    elif not cfg.data_is_pre_detrended and cfg.detrend_poly_order<0: print(f"WARNING: data_is_pre_detrended=False but detrend_poly_order={cfg.detrend_poly_order} (skip). No internal detrend. Intended?")
    main(cfg)