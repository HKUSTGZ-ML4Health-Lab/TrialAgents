"""
Run Time Series Small Model Baselines

Migrated from m-agents/baselines/run_ts_small_models.py
"""
import os, sys, json
import time
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

# TrialAgents imports
from utils.context.process_data import load_clinical_trial_datasets_old, process_patient_df_old
from utils.retrieval.hparams import nct_config as config
from utils.data_utils.paths import get_baselines_runs_path, get_baselines_saved_models_path, get_responses_path
from src.training.lnl_dataloader import process_data
from TrialAgents.utils.baselines.small_models import *

os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpuid", "0")

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class nct_time_series_dataset(torch.utils.data.Dataset):
    def __init__(self, data_zip, max_seq_len=256):
        self.data = data_zip[0]
        self.labels = data_zip[1]
        self.max_seq_len = max_seq_len

    def _process_time_series(self, patient_df):
        required_cols = ['demographic_col', 'indicator_col', 'value_col', 'time_col']
        for col in required_cols:
            if col not in patient_df.columns:
                raise ValueError(f"Missing required column: {col}")

        patient_df = patient_df.sort_values('time_col').reset_index(drop=True)
        features = patient_df[required_cols].values.astype(np.float32)

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"Warning: Found NaN or Inf in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        indicator_col = features[:, 1]
        indicator_col = np.maximum(0, indicator_col)
        indicator_col = np.floor(indicator_col)
        features[:, 1] = indicator_col

        seq_len = len(features)
        if seq_len > self.max_seq_len:
            features = features[-self.max_seq_len:]
            actual_len = self.max_seq_len
        else:
            pad_len = self.max_seq_len - seq_len
            padding = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
            features = np.vstack([padding, features])
            actual_len = seq_len

        if features.shape != (self.max_seq_len, 4):
            raise ValueError(f"Feature shape mismatch: expected ({self.max_seq_len}, 4), got {features.shape}")
        return torch.FloatTensor(features), actual_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patient_df = self.data[index]
        target = self.labels[index]
        features, seq_len = self._process_time_series(patient_df)
        if np.isnan(target) or np.isinf(target):
            target = 0.0
        target = float(target)
        target = max(0, min(1, target))
        return {
            'features': features,
            'seq_len': seq_len,
            'target': torch.FloatTensor([target])
        }


class ClassificationSemiLoss(object):
    def __init__(self, class_weights=None):
        self.class_weights = (torch.tensor(class_weights).float()) if class_weights is not None else None

    def __call__(self, outputs_x, targets_x):
        if targets_x.dim() == 1:
            Lx = F.cross_entropy(outputs_x, targets_x.long())
        else:
            log_probs_x = F.log_softmax(outputs_x, dim=1)
            if self.class_weights is not None:
                weights = self.class_weights.unsqueeze(0)
                weighted_targets_x = targets_x * weights
                Lx = -torch.mean(torch.sum(weighted_targets_x * log_probs_x, dim=1))
            else:
                Lx = -torch.mean(torch.sum(targets_x * log_probs_x, dim=1))
        return Lx


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gather_soft_labels():
    base_dir = get_responses_path()
    id2soft = {}
    for subfolder in os.listdir(base_dir):
        expected_suffix = "_morpos" if config["dataset_name"] in ['NCT03041311', 'NCT00174655'] and config.get('group') != 'control' else ""
        expected_folder = f"{config['dataset_name']}_{config['task']}_{config['llm']}_{config['corpus_name']}_{config['group']}{expected_suffix}"
        if subfolder != expected_folder:
            continue
        subfolder_path = os.path.join(base_dir, subfolder)
        for subsubfolder in os.listdir(subfolder_path):
            if subsubfolder.startswith('result'):
                continue
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            if 'supervisor_logits_conf_json.json' in os.listdir(subsubfolder_path):
                json_file = 'supervisor_logits_conf_json.json'
            else:
                json_file = 'supervisor_logits_json.json'
            json_file_path = os.path.join(subsubfolder_path, json_file)
            if not os.path.exists(json_file_path):
                continue
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if json_file == 'supervisor_logits_conf_json.json':
                    logit_key, conf_key = 'logit', 'conf'
                else:
                    logit_key, conf_key = 'final_logit', 'final_conf'
                if logit_key in data and conf_key in data:
                    data[logit_key] = 1 - data[logit_key]
                    id2soft[subsubfolder] = (data[logit_key], data[conf_key])
            except Exception as e:
                print(f"Error reading {json_file_path}: {e}")
    return id2soft


def load_data():
    dataset = config['dataset_name']
    trial_path = config['trial_path']
    data_raw = load_clinical_trial_datasets_old(trial_path=trial_path, dataset_to_process=dataset)
    data_raw = data_raw[0]
    df = pd.DataFrame(data_raw['df'])
    data_args = {
        'demographic_col': data_raw.get('demographic_col', None),
        'indicator_col': data_raw['indicator_col'],
        'value_col': data_raw['value_col'],
        'patient_id_col': data_raw['patient_id_col'],
        'time_col': data_raw['time_col'],
        'label_col': data_raw['label_col'],
    }
    patients_list = process_patient_df_old(
        df, patient_id_col=data_args['patient_id_col'],
        indicator_col=data_args['indicator_col'], value_col=data_args['value_col'],
        time_col=data_args['time_col'], label_col=data_args['label_col'],
        demographic_col=data_args['demographic_col']
    )
    return patients_list


def get_dataloaders(patients_list, config):
    kf = StratifiedKFold(n_splits=config['fold'], shuffle=True, random_state=42)
    y = [patient[-1] for patient in patients_list]
    for idx_fold, (index_train, index_test) in enumerate(kf.split(patients_list, y)):
        data_train, data_test = patients_list.iloc[index_train], patients_list.iloc[index_test]
        id2soft = gather_soft_labels()
        label_train, _ = zip(*[id2soft[str(patients_list.iloc[idx][0])] for idx in index_train])
        label_train = np.array(label_train)
        label_test = [i[-1] for i in data_test]
        data_train = [data[1:-1] for data in data_train]
        data_test = [data[1:-1] for data in data_test]
        data_train, data_valid = train_test_split(data_train, test_size=0.2, random_state=42)
        label_train, label_valid = train_test_split(label_train, test_size=0.2, random_state=42)
        train_zip, valid_zip, test_zip = process_data(
            train_zip=(data_train, label_train), es_zip=(data_valid, label_valid),
            test_zip=(data_test, label_test)
        )
        train_set = nct_time_series_dataset(train_zip, max_seq_len=64)
        valid_set = nct_time_series_dataset(valid_zip, max_seq_len=64)
        test_set = nct_time_series_dataset(test_zip, max_seq_len=64)
        yield idx_fold, (
            torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True),
            torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False),
            torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
        )


def get_metrics(y_true, y_pred_prob):
    def min_pse(labels, preds):
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, _ = precision_recall_curve(labels, preds)
        return np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

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
        _minpse = min_pse(y_true, y_pred_prob)
    except Exception:
        auprc, _minpse = float('nan'), 0
    return acc, f1, auroc, auprc, _minpse


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    train_probs, train_labels = [], []
    for batch in loader:
        features = batch['features'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        batch_data = {'features': features, 'seq_len': seq_lens}
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        train_probs.extend(probs.cpu().detach().numpy())
        train_labels.extend(torch.Tensor(targets > 0.5).long().cpu().numpy())
    train_loss /= len(loader)
    acc, f1, auroc, auprc, _ = get_metrics(train_labels, train_probs)
    return train_loss, acc, f1, auroc, auprc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_probs, valid_labels = [], []
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            targets = torch.Tensor(targets > 0.5).long().squeeze()
            seq_lens = batch['seq_len']
            batch_data = {'features': features, 'seq_len': seq_lens}
            outputs = model(batch_data)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            valid_probs.extend(probs.cpu().numpy())
            valid_labels.extend(targets.cpu().numpy())
    valid_loss /= len(loader)
    acc, f1, auroc, auprc, _ = get_metrics(valid_labels, valid_probs)
    return valid_loss, acc, f1, auroc, auprc


def test_model(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_probs, test_labels = [], []
    targets_lst, probs_lst = [], []
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device).long().squeeze()
            seq_lens = batch['seq_len']
            batch_data = {'features': features, 'seq_len': seq_lens}
            outputs = model(batch_data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(targets.cpu().numpy())
            targets_lst.extend(targets.cpu().numpy().tolist())
            probs_lst.extend(probs.cpu().numpy().tolist())
    test_loss /= len(loader)
    acc, f1, auroc, auprc, _minpse = get_metrics(test_labels, test_probs)
    preds_df = pd.DataFrame({'label': targets_lst, 'pred': probs_lst})
    return test_loss, acc, f1, auroc, auprc, _minpse, preds_df


def main():
    set_seed(config.get('seed', 42))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        patients_list = load_data()
        print(f"Successfully loaded data, patients: {len(patients_list)}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    acc_lst, auprc_lst, auroc_lst, wf1_lst, minpse_lst, preds_lst = [], [], [], [], [], []

    dataset_name = config['dataset_name']
    group = config.get('group', 'experimental')
    log_dir = get_baselines_runs_path(f'{dataset_name}_{group}_ts_models')
    today = time.strftime("%Y%m%d", time.localtime())
    writer = SummaryWriter(log_dir=os.path.join(log_dir, today))
    os.makedirs(writer.log_dir, exist_ok=True)

    for idx_fold, (train_loader, valid_loader, test_loader) in get_dataloaders(patients_list, config):
        print('=' * 50)
        print(f"Fold {idx_fold + 1} / {config['fold']}")

        # Dynamic vocab_size and seq_len
        all_indicators, all_seq_lens = [], []
        for batch in train_loader:
            indicators = batch['features'][:, :, 1].numpy().flatten()
            seq_lens_vals = batch['seq_len'].numpy()
            all_indicators.extend(indicators)
            all_seq_lens.extend(seq_lens_vals)
        all_indicators = np.array(all_indicators)
        max_indicator = int(np.ceil(all_indicators.max()))
        max_seq_len_in_data = int(np.max(all_seq_lens))
        dynamic_seq_len = max_seq_len_in_data + 1
        dynamic_vocab_size = max_indicator + 10
        print(f"Dynamic seq_len: {dynamic_seq_len}, vocab_size: {dynamic_vocab_size}")

        if dataset_name == 'NCT03041311':
            d_model = config['d_model'] // 4
            n_head = config['n_head'] // 2
        else:
            d_model = config['d_model']
            n_head = config['n_head']
        n_layer = config['n_layer']
        dropout = config['dropout']

        model_name = config['model_names'][0]
        if "Trans" in model_name:
            model = NCTTimeSeriesTransformer(
                d_model=d_model, n_heads=n_head, n_layers=n_layer, dropout=dropout,
                max_seq_len=dynamic_seq_len, indicator_vocab_size=dynamic_vocab_size, d_out=2)
        elif "LSTM" in model_name:
            model = LSTMPredictor(hidden_size=d_model // 8, num_embeddings=dynamic_vocab_size, num_layers=n_layer)
        elif "GRU" in model_name:
            model = NCTTimeSeriesGRU(
                indicator_vocab_size=dynamic_vocab_size, max_seq_len=dynamic_seq_len,
                hidden_size=32, num_layers=1, dropout=0.1, d_out=2, bidirectional=False)
        else:
            model = NCTTimeSeriesTransformer(
                d_model=d_model, n_heads=n_head, n_layers=n_layer, dropout=dropout,
                max_seq_len=dynamic_seq_len, indicator_vocab_size=dynamic_vocab_size, d_out=2)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name} created, params: {total_params:,}")
        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        pos_count = sum(1 for l in train_loader.dataset.labels if l > 0.5)
        neg_count = len(train_loader.dataset.labels) - pos_count
        total = pos_count + neg_count
        neg_weight = total / (2.0 * neg_count)
        pos_weight = total / (2.0 * pos_count)
        class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float).to(device)
        print(f"Class - Neg: {neg_count}, Pos: {pos_count}, Weights: [{neg_weight:.4f}, {pos_weight:.4f}]")

        criterion_train = ClassificationSemiLoss(class_weights=class_weights)
        criterion_test = nn.CrossEntropyLoss(weight=class_weights)

        patience = config["patience"] * 2
        best_loss = float('inf')
        best_epoch = 0

        for epoch in range(config['epochs']):
            try:
                train_loss, train_acc, train_f1, train_auroc, train_auprc = train_one_epoch(
                    model, train_loader, optimizer, criterion_train, device)
                valid_loss, valid_acc, valid_f1, valid_auroc, valid_auprc = eval_one_epoch(
                    model, valid_loader, criterion_test, device)
            except Exception as e:
                print(f"Train/eval error: {e}")
                break

            writer.add_scalar(f'Fold{idx_fold+1}/Train/Loss', train_loss, epoch)
            writer.add_scalar(f'Fold{idx_fold+1}/Valid/Loss', valid_loss, epoch)
            writer.add_scalar(f'Fold{idx_fold+1}/Valid/AUROC', valid_auroc, epoch)

            print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch + 1
                save_path = get_baselines_saved_models_path(dataset_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, f"best_ts_model_fold_{idx_fold+1}.pth"))
                patience = config["patience"] * 2
            else:
                patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

        try:
            test_loss, test_acc, test_f1, test_auroc, test_auprc, test_minpse, test_preds_df = test_model(
                model, test_loader, criterion_test, device)
        except Exception as e:
            print(f"Test error: {e}")
            continue

        print(f"Test - ACC: {test_acc:.4f}, AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, MINPSE: {test_minpse:.4f}")
        acc_lst.append(test_acc)
        auroc_lst.append(test_auroc)
        auprc_lst.append(test_auprc)
        wf1_lst.append(test_f1)
        minpse_lst.append(test_minpse)
        preds_lst.append(test_preds_df)

    all_preds_df = pd.concat(preds_lst, axis=0)
    all_preds_df.to_csv(os.path.join(log_dir, f'{config["model_names"][0]}_preds_roc={np.mean(auroc_lst):.4f}_prauc={np.mean(auprc_lst):.4f}.csv'), index=False)

    print("=" * 50)
    print("FINAL RESULTS:")
    print(f"Average ACC: {np.mean(acc_lst):.4f} ± {np.std(acc_lst):.4f}")
    print(f"Average AUROC: {np.mean(auroc_lst):.4f} ± {np.std(auroc_lst):.4f}")
    print(f"Average AUPRC: {np.mean(auprc_lst):.4f} ± {np.std(auprc_lst):.4f}")
    print(f"Average MINPSE: {np.mean(minpse_lst):.4f} ± {np.std(minpse_lst):.4f}")

    with open(os.path.join(log_dir, 'result.txt'), 'a') as f:
        f.write('\n' + '=' * 100)
        f.write(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Dataset: {dataset_name}, Task: {config['task']}, LLM: {config['llm']}, Group: {group}\n")
        f.write(f"Average ACC: {np.mean(acc_lst):.4f} ± {np.std(acc_lst):.4f}\n")
        f.write(f"Average AUROC: {np.mean(auroc_lst):.4f} ± {np.std(auroc_lst):.4f}\n")
        f.write(f"Average AUPRC: {np.mean(auprc_lst):.4f} ± {np.std(auprc_lst):.4f}\n")
        f.write(f"Average MINPSE: {np.mean(minpse_lst):.4f} ± {np.std(minpse_lst):.4f}\n")

    writer.add_scalar('Overall/AUROC_mean', np.mean(auroc_lst), 0)
    writer.close()


if __name__ == '__main__':
    set_seed(42)
    main()
