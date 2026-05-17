import os, sys
import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision
from torchmetrics.classification import BinaryF1Score
import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, balanced_accuracy_score

def get_final_result(logits):
    # 获取各医生的logit和置信度权重
    clinician_logits = [logits['clinician0_logit'], logits['clinician1_logit'], 
                    logits['clinician2_logit'], logits['meta_logit']]
    clinician_confs = [logits['clinician0_conf'], logits['clinician1_conf'], 
                   logits['clinician2_conf'], logits['meta_conf']]
    
    # 转换置信度为权重
    # conf_weights = [trans_confidence(conf) for conf in clinician_confs]
    conf_weights = [conf for conf in clinician_confs]
    
    # 计算加权平均的final_logit
    weighted_sum = sum(logit * weight for logit, weight in zip(clinician_logits, conf_weights))
    total_weight = sum(conf_weights)
    
    # 归一化final_logit到0-1范围
    final_logit = weighted_sum / total_weight if total_weight > 0 else 0.5
    
    # 归一化final_conf到0-1范围
    total_conf = sum(clinician_confs)
    final_conf = total_conf / len(clinician_confs) if len(clinician_confs) > 0 else 0.5
    
    # 确保结果在0-1范围内
    final_logit = max(0.0, min(1.0, final_logit))
    final_conf = max(0.0, min(1.0, final_conf))
    # 保留四位小数
    final_logit = round(final_logit, 4)
    final_conf = round(final_conf, 4)
    
    return final_logit, final_conf

def trans_confidence(x):
    x = float(x)
    if x <= 0.6: return 0.1
    if 0.8 > x > 0.6: return 0.3
    if 0.9 > x >= 0.8: return 0.5
    if 1 > x >= 0.9: return 0.8
    if x == 1: return 1

def metrics(pat_id_lst, predictions, probs, labels, save_dir='./'):
    # evaluation
    predictions = np.array([predictions[i] for i in range(len(predictions))])
    probs = np.array([probs[i] for i in range(len(probs))])
    rocauc = roc_auc_score(labels, probs)
    prauc = average_precision_score(labels, probs)
    w_f1 = f1_score(labels, predictions, average='weighted') 
    balanced_acc = balanced_accuracy_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions)
    # rocauc_list.append(rocauc)
    # prauc_list.append(prauc)
    # f1_list.append(f1)

    preds_df = pd.DataFrame({'label': labels, 'pred': predictions})
    preds_df['pat_id'] = pat_id_lst
    preds_df = preds_df.set_index('pat_id')
    preds_df = preds_df[preds_df['label'] != preds_df['pred']].to_dict(orient='index')

    # Print results
    print("ROC AUC:", rocauc, "\tPR AUC:", prauc, "\tWeighted F1 Score:", w_f1, "\tBalanced ACC:", balanced_acc)  
    print("Diff Predictions:\n", preds_df)
    # save to txt
    with open(f'{save_dir}/results.txt', 'a') as f:
        f.write(f"ROC AUC: {rocauc}\tPR AUC: {prauc}\n")
        f.write(f"Weighted F1 Score: {w_f1}\tBalanced ACC: {balanced_acc}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Diff Predictions:\n{preds_df}\n")
        f.write(f"=" * 50 + "\n")
    # average
    # avg_rocauc, avg_prauc, avg_f1 = np.mean(rocauc_list), np.mean(prauc_list), np.mean(f1_list)
    # std_rocauc, std_prauc, std_f1 = np.std(rocauc_list), np.std(prauc_list), np.std(f1_list) 
    # print(f"Average ROC AUC: {avg_rocauc}±{std_rocauc},\tAverage PR AUC:{avg_prauc}±{std_prauc},\tAverage F1 Score: {avg_f1}±{std_f1}")
    # with open('results.txt', 'a') as f:
    #     f.write(f"Average ROC AUC: {avg_rocauc}±{std_rocauc},\tAverage PR AUC:{avg_prauc}±{std_prauc},\tAverage F1 Score: {avg_f1}±{std_f1}")


def minpse(preds, labels):
    precisions, recalls, _ = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score


def get_binary_metrics(preds, labels):
    """Calculate binary classification metrics safely handling both tensors and arrays"""
    try:
        # Use sklearn metrics directly to avoid tensor compatibility issues
        from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
        
        # Ensure inputs are numpy arrays
        if hasattr(preds, 'detach'):  # If it's a tensor
            preds_np = preds.detach().cpu().numpy()
        else:
            preds_np = np.array(preds, dtype=np.float64)
            
        if hasattr(labels, 'detach'):  # If it's a tensor
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.array(labels, dtype=np.float64)
        
        # Convert predictions to binary for accuracy and f1
        preds_binary = (preds_np > 0.5).astype(int)
        labels_int = labels_np.astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels_int, preds_binary)
        
        # Check if we have both classes for AUC calculation
        if len(np.unique(labels_int)) > 1:
            auroc = roc_auc_score(labels_int, preds_np)
            auprc = average_precision_score(labels_int, preds_np)
        else:
            auroc = 0.5  # Random performance when only one class
            auprc = np.mean(labels_int)  # Baseline for single class
        
        f1 = f1_score(labels_int, preds_binary)
        minpse_score = minpse(preds_np, labels_np)
        
        return {
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse_score,
            "f1": f1,
        }
        
    except Exception as e:
        print(f"Error in get_binary_metrics: {e}")
        # Return default values if calculation fails
        return {
            "accuracy": 0.0,
            "auroc": 0.5,
            "auprc": 0.0,
            "minpse": 0.0,
            "f1": 0.0,
        }


def check_metric_is_better(cur_best, score, main_metric='auroc'):
    if cur_best == {}:
        return True
    if score > cur_best[main_metric]:
        return True
    return False


def bootstrap(preds, labels, K=100, seed=42):
    """Bootstrap resampling for binary classification metrics. Resample K times"""
    
    # Convert to numpy arrays if they aren't already
    preds = np.array(preds, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    
    length = len(preds)
    # length = N
    np.random.seed(seed)
    
    # Initialize a list to store bootstrap samples
    bootstrapped_samples = []

    # Create K bootstrap samples
    for _ in range(K):
        # Sample with replacement from the indices
        sample_indices = np.random.choice(length, length, replace=True)

        # Use the sampled indices to get the bootstrap sample of preds and labels
        sample_preds = preds[sample_indices]
        sample_labels = labels[sample_indices]
        
        # Store the bootstrap samples
        bootstrapped_samples.append((sample_preds, sample_labels))

    return bootstrapped_samples


def export_metrics(bootstrapped_samples):
    """Export metrics from bootstrap samples with robust error handling"""
    metrics = {"accuracy": [], "auprc": [], "auroc": [], "minpse": [], "f1": []}
    
    for i, sample in enumerate(bootstrapped_samples):
        try:
            sample_preds, sample_labels = sample[0], sample[1]
            res = get_binary_metrics(sample_preds, sample_labels)

            for k, v in res.items():
                metrics[k].append(v)
                
        except Exception as e:
            print(f"Warning: Bootstrap sample {i} failed: {e}")
            continue

    # Convert to numpy array and calculate statistics
    final_metrics = {}
    for k, v in metrics.items():
        if v:  # Only process if we have valid values
            values_array = np.array(v)
            final_metrics[k] = {"mean": np.mean(values_array), "std": np.std(values_array)}
        else:
            # If no valid values, return defaults
            final_metrics[k] = {"mean": 0.0, "std": 0.0}
    
    return final_metrics


def run_bootstrap(preds, labels, K=100, seed=42):
    bootstrap_samples = bootstrap(preds, labels, K=K, seed=seed)
    metrics = export_metrics(bootstrap_samples)
    return metrics