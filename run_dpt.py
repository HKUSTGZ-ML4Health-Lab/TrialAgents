"""
LNL Training Entry Point for TrialAgents

VCTC (Virtual Clinical Trial Center) class orchestrates:
1. Read LLM-generated soft labels from response directories
2. Pretrain small models (Transformer/LSTM/GRU) with DivideMix on soft labels
3. Cross-validation evaluation

Usage:
    python run_dpt.py
"""
import random
import os, sys, json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from tqdm import tqdm

from utils.retrieval.hparams import nct_config as config
from utils.data_utils.paths import get_responses_path, get_training_runs_path, get_logs_path
from utils.context.process_data import load_clinical_trial_datasets_old, process_patient_df_old
from utils.metrics import minpse
from src.training.lnl_train import run_dividemix
from baselines.small_models import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VCTC:
    """Virtual Clinical Trial Center - orchestrates soft labeling and small model training."""

    def __init__(self, tau=0.9, device='cuda'):
        self.tau = tau
        self.device = device

    def _preprocess_data(self, patients_list, max_length=None, num_embeddings=None,
                         events2ind=None, ind2events=None, mean_time=None, std_time=None):
        for i in range(len(patients_list)):
            patients_list[i][2].append('[END]')
            patients_list[i][3].append(patients_list[i][3][-1])
            patients_list[i][4].append(patients_list[i][4][-1])

        if max_length is None:
            max_length = max([len(patients_list[i][2]) for i in range(len(patients_list))])

        unique_events = np.unique([_ for i in range(len(patients_list)) for _ in patients_list[i][2]])
        if num_embeddings is None:
            num_embeddings = len(unique_events)
        if events2ind is None:
            events2ind = {unique_events[i]: i for i in range(len(unique_events))}
        if ind2events is None:
            ind2events = {v: k for k, v in events2ind.items()}
        if mean_time is None:
            mean_time = np.mean([np.mean(patients_list[i][4]) for i in range(len(patients_list))])
            std_time = np.mean([np.std(patients_list[i][4]) for i in range(len(patients_list))])

        for i in range(len(patients_list)):
            patients_list[i][2] = torch.Tensor(
                [events2ind[_] + 1 for _ in patients_list[i][2]] +
                [0] * (max_length - len(patients_list[i][2]))
            ).long()
            patients_list[i][3] = torch.Tensor(
                patients_list[i][3] + [0] * (max_length - len(patients_list[i][3])))
            patients_list[i][4] = torch.Tensor(
                patients_list[i][4] + [0] * (max_length - len(patients_list[i][4])))
            patients_list[i][4] = patients_list[i][4] / std_time

        print('num_patient', len(patients_list), 'max_length', max_length)
        return patients_list, max_length, num_embeddings, events2ind, ind2events, mean_time, std_time

    def gather_soft_labels(self):
        base_dir = get_responses_path()
        soft_labels, conf = [], []
        id2soft = {}
        for subfolder in os.listdir(base_dir):
            expected_suffix = "_morpos" if (
                config["dataset_name"] in ['NCT03041311', 'NCT00174655', 'NCT00079274']
                and config.get('group') != 'control'
            ) else ""
            expected_folder = (
                f"{config['dataset_name']}_{config['task']}_{config['llm']}_"
                f"{config['corpus_name']}_{config['group']}{expected_suffix}"
            )
            if subfolder != expected_folder:
                continue
            subfolder_path = os.path.join(base_dir, subfolder)
            for subsubfolder in os.listdir(subfolder_path):
                if subsubfolder.startswith('result'):
                    continue
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)

                json_file = ('supervisor_logits_conf_json.json'
                             if 'supervisor_logits_conf_json.json' in os.listdir(subsubfolder_path)
                             else 'supervisor_logits_json.json')

                json_file_path = os.path.join(subsubfolder_path, json_file)
                if not os.path.exists(json_file_path):
                    continue

                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    logit_key = 'logit' if json_file == 'supervisor_logits_conf_json.json' else 'final_logit'
                    conf_key = 'conf' if json_file == 'supervisor_logits_conf_json.json' else 'final_conf'

                    if logit_key in data and conf_key in data:
                        soft_labels.append(data[logit_key])
                        conf.append(data[conf_key])
                        id2soft[subsubfolder] = (data[logit_key], data[conf_key])
                except Exception as e:
                    print(f"Error reading {json_file_path}: {e}")
        return np.array(soft_labels), np.array(conf), id2soft

    def soft_llm_labeling(self):
        y_soft, y_conf, id2soft = self.gather_soft_labels()
        return y_soft, y_conf, id2soft

    def train_small_model(self, y_soft, y_conf, id2soft):
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
            indicator_col=data_args['indicator_col'],
            value_col=data_args['value_col'],
            time_col=data_args['time_col'],
            label_col=data_args['label_col'],
            demographic_col=data_args['demographic_col']
        )

        preprocess_result = self._preprocess_data(patients_list)
        patients_list = preprocess_result[0]

        act_len = min(len(patients_list), len(y_soft))
        patients_list, y_soft = patients_list.iloc[:act_len], y_soft[:act_len]

        print(f"Total patients: {len(patients_list)}")

        group = config.get('group', 'experimental')
        llm = config.get('llm', 'default')
        log_dir = get_training_runs_path(f'{dataset}_{llm}_{group}')
        if dataset in ['NCT03041311', 'NCT00174655', 'NCT00079274'] and group != 'control':
            log_dir += '_morpos'
        os.makedirs(log_dir, exist_ok=True)

        acc_lst, auprc_lst, auroc_lst, f1_lst, minpse_lst, preds_lst = [], [], [], [], [], []

        kf = StratifiedKFold(n_splits=config['fold'], shuffle=True, random_state=42)
        y = [patient[-1] for patient in patients_list]
        for idx_fold, (index_train, index_test) in enumerate(kf.split(patients_list, y)):
            print(f"Fold {idx_fold + 1} / {config['fold']}")
            data_train, label_train, conf_train = [], [], []
            for idx in index_train:
                if str(patients_list.iloc[idx][0]) in id2soft:
                    data_train.append(patients_list.iloc[idx])
                    label_train.append(id2soft[str(patients_list.iloc[idx][0])][0])
                    conf_train.append(id2soft[str(patients_list.iloc[idx][0])][1])
            data_train = pd.Series(data_train)
            label_train = np.array(label_train)
            conf_train = np.array(conf_train)
            data_test = patients_list.iloc[index_test]
            label_test = [i[-1] for i in data_test]

            # Label inversion logic (preserved from original)
            if 'deepseek' in log_dir:
                if '_morpos' in log_dir:
                    if 'NCT03041311' in dataset or 'NCT00174655' in dataset:
                        pass
                    else:
                        label_test = [1 - x for x in label_test]
                else:
                    if 'NCT00694382' in dataset:
                        label_test = [1 - x for x in label_test]
                        label_train = [1 - x for x in label_train]
                    else:
                        label_train = [1 - x for x in label_train]
                    if 'NCT00041119' in dataset or 'NCT00694382' in dataset:
                        conf_train = np.array([1 - x for x in conf_train])

            train_cnt = sum(1 for x in label_train if x > 0.5)
            test_cnt = sum(1 for x in label_test if x > 0.5)
            print(f"label_train pos rate: {train_cnt}/{len(label_train)}")
            print(f"label_test pos rate: {test_cnt}/{len(label_test)}")

            es_idx = (conf_train >= self.tau)
            data_es = data_train[es_idx]
            label_train_array = np.array(label_train)
            label_es = label_train_array[es_idx].tolist()

            if len(data_es) == 0:
                print(f"Warning: No early stopping samples (tau={self.tau}), using first sample")
                data_es = data_train.iloc[:1]
                label_es = [label_train[0]]

            label_es = [1 if y >= 0.5 else 0 for y in label_es]
            train_zip = [data_train, label_train]
            es_zip = [data_es, label_es]
            test_zip = [data_test, label_test]

            fold_result = run_dividemix(
                log_dir=log_dir, fold=idx_fold + 1,
                train_zip=train_zip, es_zip=es_zip, test_zip=test_zip
            )

            acc_lst.append(fold_result['acc'])
            f1_lst.append(fold_result['f1'])
            auprc_lst.append(fold_result['auprc'])
            auroc_lst.append(fold_result['auroc'])
            minpse_lst.append(fold_result['minpse'])
            preds_lst.append(fold_result['preds_df'])

        print(f"Average ACC: {np.mean(acc_lst):.4f} ± {np.std(acc_lst):.4f}")
        print(f"Average F1: {np.mean(f1_lst):.4f} ± {np.std(f1_lst):.4f}")
        print(f"Average AUPRC: {np.mean(auprc_lst):.4f} ± {np.std(auprc_lst):.4f}")
        print(f"Average AUROC: {np.mean(auroc_lst):.4f} ± {np.std(auroc_lst):.4f}")
        print(f"Average MINPSE: {np.mean(minpse_lst):.4f} ± {np.std(minpse_lst):.4f}")

        all_preds_df = pd.concat(preds_lst, axis=0)
        model_name = config.get('model_names', ['Transformer'])[0]
        all_preds_df.to_csv(
            os.path.join(log_dir,
                         f'{model_name}_preds_roc={np.mean(auroc_lst):.4f}_prauc={np.mean(auprc_lst):.4f}.csv'),
            index=False
        )

        with open(os.path.join(log_dir, 'result.txt'), 'a') as f:
            f.write('\n' + '=' * 100)
            f.write(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write(f"Dataset: {dataset}, Task: {config['task']}, LLM: {llm}, "
                    f"Corpus: {config['corpus_name']}, Group: {group}\n")
            f.write(f"LR: {config['learning_rate']}, Tau: {self.tau}, "
                    f"Batch: {config['batch_size']}, Epochs: {config['epochs']}, "
                    f"Patience: {config['patience']}, Temp: {config['T']}, WD: {config['weight_decay']}\n")
            f.write(f"Model: {model_name}, d_model: {config['d_model']}, "
                    f"n_head: {config['n_head']}, n_layer: {config['n_layer']}, dropout: {config['dropout']}\n")
            for i in range(len(acc_lst)):
                f.write(f"Fold {i + 1}: ACC: {acc_lst[i]:.4f}, F1: {f1_lst[i]:.4f}, "
                        f"AUPRC: {auprc_lst[i]:.4f}, AUROC: {auroc_lst[i]:.4f}\n")
            f.write(f"\nAverage: ACC: {np.mean(acc_lst):.4f} ± {np.std(acc_lst):.4f}\n")
            f.write(f"Average: F1: {np.mean(f1_lst):.4f} ± {np.std(f1_lst):.4f}\n")
            f.write(f"Average: AUPRC: {np.mean(auprc_lst):.4f} ± {np.std(auprc_lst):.4f}\n")
            f.write(f"Average: AUROC: {np.mean(auroc_lst):.4f} ± {np.std(auroc_lst):.4f}\n")
            f.write(f"Average: MINPSE: {np.mean(minpse_lst):.4f} ± {np.std(minpse_lst):.4f}\n")

    def reverse_tune_llm(self, X):
        pass

    def fit(self, iterations=3):
        for t in tqdm(range(iterations), desc="Iterations"):
            y_soft, y_conf, id2soft = self.soft_llm_labeling()
            self.train_small_model(y_soft, y_conf, id2soft)
        print("Training complete.")


def random_guess():
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
        indicator_col=data_args['indicator_col'],
        value_col=data_args['value_col'],
        time_col=data_args['time_col'],
        label_col=data_args['label_col'],
        demographic_col=data_args['demographic_col']
    )
    y_true = np.array([patient[-1] for patient in patients_list])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auroc_list, auprc_list, minpse_list = [], [], []
    rng = np.random.RandomState(42)

    for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(y_true)), y_true), 1):
        y_test = y_true[test_idx]
        y_pred = rng.random(size=len(test_idx))
        if len(np.unique(y_test)) > 1:
            auroc = roc_auc_score(y_test, y_pred)
        else:
            auroc = float('nan')
        auprc = average_precision_score(y_test, y_pred)
        mp = minpse(y_pred, y_test)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
        minpse_list.append(mp)
        print(f"Fold {fold} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, MINPSE: {mp:.4f}")

    print(f"Average AUROC: {np.nanmean(auroc_list):.4f} ± {np.nanstd(auroc_list):.4f}")
    print(f"Average AUPRC: {np.mean(auprc_list):.4f} ± {np.std(auprc_list):.4f}")
    print(f"Average MINPSE: {np.mean(minpse_list):.4f} ± {np.std(minpse_list):.4f}")

    y_pred_all = rng.random(size=len(y_true))
    overall_auroc = roc_auc_score(y_true, y_pred_all)
    overall_auprc = average_precision_score(y_true, y_pred_all)
    overall_minpse = minpse(y_pred_all, y_true)
    print(f"Overall - AUROC: {overall_auroc:.4f}, AUPRC: {overall_auprc:.4f}, MINPSE: {overall_minpse:.4f}")


if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    # Uncomment to run full training:
    # vctc = VCTC(tau=config.get('tau', 0.75))
    # vctc.fit(iterations=config.get('loop_num', 1))
    random_guess()
