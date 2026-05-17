"""
Run Tabular Small Model Baselines

Migrated from m-agents/baselines/run_tabular_small_models.py
"""
import os, sys, json
import time
import random
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

# TrialAgents imports
from utils.context.process_data import load_clinical_trial_datasets_old, process_patient_df_old
from utils.retrieval.hparams import nct_config as config
from utils.data_utils.paths import get_baselines_runs_path, get_baselines_saved_models_path
from src.training.lnl_dataloader import process_data
from utils.baselines.small_models import *

os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpuid", "0")

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class nct_dataset(torch.utils.data.Dataset):
    def __init__(self, data_zip):
        self.data = data_zip[0]
        self.labels = data_zip[1]

    def _dataframe_to_tensor(self, df):
        features = df[['demographic_col', 'indicator_col', 'value_col', 'time_col']].values
        flattened = features.flatten()
        flattened = np.nan_to_num(flattened)
        if len(flattened) < 100:
            flattened = np.pad(flattened, (0, 100 - len(flattened)), mode='constant')
        elif len(flattened) > 100:
            flattened = flattened[:100]
        return torch.FloatTensor(flattened)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patient_df = self.data[index]
        target = self.labels[index]
        features = self._dataframe_to_tensor(patient_df)
        return features, target


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    dataset = config['dataset_name']
    trial_path = config['trial_path']
    data_raw = load_clinical_trial_datasets_old(trial_path=trial_path, dataset_to_process=dataset)
    data_raw = data_raw[0]
    df = pd.DataFrame(data_raw['df'])
    data_args = {
        'demographic_col': data_raw['demographic_col'],
        'indicator_col': data_raw['indicator_col'],
        'value_col': data_raw['value_col'],
        'patient_id_col': data_raw['patient_id_col'],
        'time_col': data_raw['time_col'],
        'label_col': data_raw['label_col'],
    }
    patients_list = process_patient_df_old(
        df, patient_id_col=data_args['patient_id_col'],
        demographic_col=data_args['demographic_col'],
        indicator_col=data_args['indicator_col'], value_col=data_args['value_col'],
        time_col=data_args['time_col'], label_col=data_args['label_col'],
    )
    return patients_list


def get_dataloaders(patients_list, config):
    kf = KFold(n_splits=config['fold'], shuffle=True)
    for idx_fold, (index_train, index_test) in enumerate(kf.split(patients_list)):
        data_train, data_test = patients_list.iloc[index_train], patients_list.iloc[index_test]
        label_train, label_test = [i[-1] for i in data_train], [i[-1] for i in data_test]
        data_train, data_valid = train_test_split(data_train, test_size=0.2, random_state=42)
        label_train, label_valid = train_test_split(label_train, test_size=0.2, random_state=42)
        train_zip, valid_zip, test_zip = [data_train, label_train], [data_valid, label_valid], [data_test, label_test]
        train_zip, valid_zip, test_zip = process_data(train_zip=train_zip, es_zip=valid_zip, test_zip=test_zip)
        train_set = nct_dataset(train_zip)
        valid_set = nct_dataset(valid_zip)
        test_set = nct_dataset(test_zip)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
        yield idx_fold, train_loader, valid_loader, test_loader


def get_metrics(y_true, y_pred_prob):
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_pred_prob)
    except Exception:
        auroc = float('nan')
    try:
        auprc = average_precision_score(y_true, y_pred_prob)
    except Exception:
        auprc = float('nan')
    return acc, f1, auroc, auprc


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        probs = torch.sigmoid(outputs)
        train_preds.extend(probs.cpu().detach().numpy())
        train_labels.extend(labels.cpu().numpy())
    train_loss /= len(loader)
    acc, f1, auroc, auprc = get_metrics(train_labels, train_preds)
    return train_loss, acc, f1, auroc, auprc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_preds, valid_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device).float()
            outputs = model(features)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            probs = torch.sigmoid(outputs)
            valid_preds.extend(probs.cpu().numpy())
            valid_labels.extend(labels.cpu().numpy())
    valid_loss /= len(loader)
    acc, f1, auroc, auprc = get_metrics(valid_labels, valid_preds)
    return valid_loss, acc, f1, auroc, auprc


def test_model(idx_fold, model, loader, criterion, device):
    save_path = get_baselines_saved_models_path(config['dataset_name'])
    model_path = os.path.join(save_path, f"best_tab_model_fold_{idx_fold+1}.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None, None, None, None, None
    model.eval()
    test_loss = 0.0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device).float()
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            probs = torch.sigmoid(outputs)
            test_preds.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    test_loss /= len(loader)
    acc, f1, auroc, auprc = get_metrics(test_labels, test_preds)
    return test_loss, acc, f1, auroc, auprc


def main():
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name = config['dataset_name']
    group = config.get('group', 'experimental')
    today = time.strftime("%Y%m%d", time.localtime())
    log_dir = get_baselines_runs_path(f'{dataset_name}_{group}_tab_models', today)
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(writer.log_dir, exist_ok=True)

    try:
        patients_list = load_data()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    acc_lst, auprc_lst, auroc_lst, wf1_lst = [], [], [], []
    for idx_fold, train_loader, valid_loader, test_loader in get_dataloaders(patients_list, config):
        print('=' * 50)
        print(f"Fold {idx_fold + 1} / {config['fold']}")
        model = FTTransformer_o(n_cont_features=100, d_out=1)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=5e-4)

        pos_count = sum((l == 1).sum().item() for _, l in train_loader)
        neg_count = sum((l == 0).sum().item() for _, l in train_loader)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float).to(device)
        print(f"Class - Neg: {neg_count}, Pos: {pos_count}, pos_weight: {pos_weight.item():.4f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        patience = config["patience"]
        best_loss = float('inf')

        for epoch in range(config['epochs']):
            try:
                train_loss, train_acc, train_f1, train_auroc, train_auprc = train_one_epoch(
                    model, train_loader, optimizer, criterion, device)
                valid_loss, valid_acc, valid_f1, valid_auroc, valid_auprc = eval_one_epoch(
                    model, valid_loader, criterion, device)
            except Exception as e:
                print(f"Train/eval error: {e}")
                break

            writer.add_scalar(f'Fold{idx_fold+1}/Train/Loss', train_loss, epoch)
            writer.add_scalar(f'Fold{idx_fold+1}/Valid/Loss', valid_loss, epoch)
            print(f"Epoch {epoch+1}/{config['epochs']} - Train: {train_loss:.4f}, Valid: {valid_loss:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                save_path = get_baselines_saved_models_path(dataset_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, f"best_tab_model_fold_{idx_fold+1}.pth"))
                patience = config["patience"]
            else:
                patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

        try:
            test_loss, test_acc, test_auroc, test_auprc, test_f1 = test_model(
                idx_fold, model, test_loader, criterion, device)
        except Exception as e:
            print(f"Test error: {e}")
            continue

        print(f"Test - ACC: {test_acc:.4f}, AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}")
        acc_lst.append(test_acc)
        auroc_lst.append(test_auroc)
        auprc_lst.append(test_auprc)
        wf1_lst.append(test_f1)

    print(f"Average ACC: {np.mean(acc_lst):.4f} ± {np.std(acc_lst):.4f}")
    print(f"Average AUROC: {np.mean(auroc_lst):.4f} ± {np.std(auroc_lst):.4f}")
    print(f"Average AUPRC: {np.mean(auprc_lst):.4f} ± {np.std(auprc_lst):.4f}")
    writer.close()


if __name__ == '__main__':
    main()
