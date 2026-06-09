"""
NCT Data Loader for DivideMix Training

Migrated from m-agents/utils/lnl_dataloader.py
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import torch
from torchnet.meter import AUCMeter

import warnings
warnings.filterwarnings("ignore")


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def process_data(train_zip, es_zip, test_zip):
    """
    Process nested list structure data.
    Each sample format: [indicator_col_list, value_col_list, time_col_list]
    """
    X_train, y_train = train_zip[0], train_zip[1]
    X_es, y_es = es_zip[0], es_zip[1]
    X_test, y_test = test_zip[0], test_zip[1]

    def process_patient_data(X_series, y_labels):
        processed_data = []
        for i, patient_data in enumerate(X_series):
            demographic_list, indicator_list, value_list, time_list = (
                patient_data[1], patient_data[2], patient_data[3], patient_data[4])

            num_events = len(indicator_list)
            demo = list(demographic_list)
            demo_matrix = np.tile(demo, (num_events, 1))
            demo_df = pd.DataFrame(demo_matrix, columns=[f"demo_{i}" for i in range(len(demo))])

            event_df = pd.DataFrame({
                "indicator_col": indicator_list,
                "value_col": value_list,
                "time_col": time_list
            })

            df = pd.concat([demo_df, event_df], axis=1)

            # df = pd.DataFrame({
            #     'demographic_cols': demographic_list,
            #     'indicator_col': indicator_list,
            #     'value_col': value_list,
            #     'time_col': time_list
            # })

            processed_data.append(df)
        return processed_data, y_labels

    def encode_features(train_data_list, es_data_list, test_data_list, train_labels):
        all_indicators = []
        for patient_df in train_data_list:
            all_indicators.extend(patient_df['indicator_col'].tolist())
        for patient_df in es_data_list:
            all_indicators.extend(patient_df['indicator_col'].tolist())
        for patient_df in test_data_list:
            all_indicators.extend(patient_df['indicator_col'].tolist())

        unique_indicators = list(set(all_indicators))
        from sklearn.preprocessing import LabelEncoder
        cat_encoder = LabelEncoder()
        cat_encoder.fit(unique_indicators)

        def encode_patient_data(data_list):
            encoded_data = []
            for patient_df in data_list:
                df_encoded = patient_df.copy()
                df_encoded['indicator_col'] = cat_encoder.transform(patient_df['indicator_col'])
                df_encoded['value_col'] = pd.to_numeric(df_encoded['value_col'], errors='coerce').fillna(0)
                df_encoded['time_col'] = pd.to_numeric(df_encoded['time_col'], errors='coerce').fillna(0)
                demo_cols = [c for c in df_encoded.columns if c.startswith("demo_")]
                df_encoded[demo_cols] = df_encoded[demo_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0)
                encoded_data.append(df_encoded)
            return encoded_data

        train_encoded = encode_patient_data(train_data_list)
        es_encoded = encode_patient_data(es_data_list)
        test_encoded = encode_patient_data(test_data_list)
        return train_encoded, es_encoded, test_encoded, cat_encoder

    train_data_list, train_labels = process_patient_data(X_train, y_train)
    es_data_list, es_labels = process_patient_data(X_es, y_es)
    test_data_list, test_labels = process_patient_data(X_test, y_test)

    train_encoded, es_encoded, test_encoded, cat_encoder = encode_features(
        train_data_list, es_data_list, test_data_list, train_labels
    )

    return (train_encoded, train_labels), (es_encoded, es_labels), (test_encoded, test_labels)


class MedicalTimeSeriesAugmentation:
    """Time series data augmentation for medical data."""
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def temporal_jitter(self, df, time_col='time_col', jitter_std=0.1):
        augmented_df = df.copy()
        if time_col in df.columns:
            time_noise = np.random.normal(0, jitter_std, len(df))
            augmented_df[time_col] = np.maximum(0, df[time_col] + time_noise)
        return augmented_df

    def event_dropout(self, df, dropout_prob=0.1):
        if len(df) <= 1:
            return df
        keep_mask = np.random.random(len(df)) > dropout_prob
        if not keep_mask.any():
            keep_mask[0] = True
        return df[keep_mask].reset_index(drop=True)

    def event_duplication(self, df, dup_prob=0.1):
        augmented_df = df.copy()
        dup_mask = np.random.random(len(df)) < dup_prob
        if dup_mask.any():
            dup_events = df[dup_mask].copy()
            time_noise = np.random.normal(0, 0.01, len(dup_events))
            dup_events['time_col'] = np.maximum(0, dup_events['time_col'] + time_noise)
            augmented_df = pd.concat([augmented_df, dup_events], ignore_index=True)
            augmented_df = augmented_df.sort_values('time_col').reset_index(drop=True)
        return augmented_df

    def value_perturbation(self, df, value_col='value_col', noise_ratio=0.05):
        augmented_df = df.copy()
        if value_col in df.columns:
            values = df[value_col].values
            noise = np.random.normal(0, noise_ratio * np.std(values), len(values))
            augmented_df[value_col] = np.maximum(0, values + noise)
        return augmented_df

    def sequence_shuffle(self, df, shuffle_prob=0.1):
        if len(df) <= 2:
            return df
        augmented_df = df.copy()
        if np.random.random() < shuffle_prob:
            window_size = min(3, len(df))
            start_idx = np.random.randint(0, len(df) - window_size + 1)
            end_idx = start_idx + window_size
            window_data = augmented_df.iloc[start_idx:end_idx].copy()
            shuffled_indices = np.random.permutation(len(window_data))
            augmented_df.iloc[start_idx:end_idx] = window_data.iloc[shuffled_indices].values
        return augmented_df


class nct_dataset(Dataset):
    def __init__(self, mode, train_zip, es_zip, test_zip, pred=None, probability=None, log=None,
                 max_seq_len=64, use_soft_labels=False, soft_label_method=None, soft_label_config=None):
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
        self.max_seq_len = max_seq_len

        self.use_soft_labels = use_soft_labels
        self.soft_label_method = soft_label_method
        self.soft_label_config = soft_label_config or {
            'confidence_threshold': 0.7,
            'smoothing_epsilon': 0.1,
            'temperature': 1.0,
        }

        self.augmentation = MedicalTimeSeriesAugmentation()
        self.aug_config = {
            'use_temporal_jitter': True,
            'use_event_dropout': False,
            'use_event_duplication': False,
            'use_value_perturbation': True,
            'use_sequence_shuffle': False,
        }

        if self.mode == 'test':
            self.test_data = test_zip[0]
            self.test_label = test_zip[1]
        elif self.mode == 'eval':
            self.es_data = es_zip[0]
            self.es_label = es_zip[1]
        else:
            train_data, train_label = train_zip[0], train_zip[1]
            self.train_data = train_data
            self.train_label = train_label
            noise_label = train_label
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    if len(pred_idx) == 0:
                        print("Warning: No labeled samples found, using first sample as labeled")
                        pred_idx = [0] if len(train_data) > 0 else []
                    self.probability = [probability[i] for i in pred_idx] if len(pred_idx) > 0 else [0.5]
                    clean = (np.array(noise_label) == np.array(train_label))
                    try:
                        if len(self.probability) > 0 and len(clean) > 0:
                            clean_selected = clean[pred_idx] if len(pred_idx) > 0 else [1]
                            unique_labels = np.unique(clean_selected)
                            if len(unique_labels) < 2:
                                log.write(f'Warning: Only one class present: {unique_labels}\n') if log else None
                                auc = 0.5
                            else:
                                auc_meter = AUCMeter()
                                auc_meter.reset()
                                auc_meter.add(self.probability, clean_selected)
                                auc, _, _ = auc_meter.value()
                                if np.isnan(auc):
                                    log.write('Warning: AUC calculation returned NaN\n') if log else None
                                    auc = 0.5
                        else:
                            auc = 0.5
                    except Exception:
                        auc = 0.5
                    if log:
                        log.write('Number of labeled samples:%d   AUC:%.3f\n' % (len(pred_idx), auc))
                        log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                    if len(pred_idx) == 0:
                        print("Warning: No unlabeled samples found, using last sample as unlabeled")
                        pred_idx = [len(train_data) - 1] if len(train_data) > 0 else []

                if len(pred_idx) > 0:
                    self.train_data = [train_data[i] for i in pred_idx]
                    self.noise_label = [noise_label[i] for i in pred_idx]
                else:
                    self.train_data = [train_data[0]] if len(train_data) > 0 else []
                    self.noise_label = [noise_label[0]] if len(noise_label) > 0 else []

                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def _process_time_series(self, patient_df):
        demo_cols = [c for c in patient_df.columns if c.startswith("demo_")]
        if not demo_cols:
            raise ValueError("Missing demo_ columns")

        def _demo_key(c):
            try:
                return int(c.split("_", 1)[1])
            except Exception:
                return c

        demo_cols = sorted(demo_cols, key=_demo_key)
        required_cols = demo_cols + ["indicator_col", "value_col", "time_col"]

        missing = [c for c in required_cols if c not in patient_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        patient_df = patient_df.sort_values("time_col").reset_index(drop=True)
        features = patient_df[required_cols].values.astype(np.float32)
        seq_len = len(features)

        if seq_len > self.max_seq_len:
            features = features[-self.max_seq_len:]
            actual_len = self.max_seq_len
        else:
            pad_len = self.max_seq_len - seq_len
            padding = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
            features = np.vstack([features, padding])
            actual_len = seq_len

        return torch.FloatTensor(features), actual_len

    def __getitem__(self, index):
        if self.mode == 'labeled':
            patient_df = self.train_data[index]
            target = self.noise_label[index]
            prob = self.probability[index]
            soft_target = target

            aug_data1, aug_data2 = patient_df, patient_df
            features1, seq_len1 = self._process_time_series(aug_data1)
            features2, seq_len2 = self._process_time_series(aug_data2)

            return {
                'features1': features1, 'features2': features2,
                'prob': prob, 'seq_len1': seq_len1, 'seq_len2': seq_len2,
                'target': soft_target, 'index': index
            }

        elif self.mode == 'unlabeled':
            patient_df = self.train_data[index]
            aug_data1, aug_data2 = patient_df, patient_df
            features1, seq_len1 = self._process_time_series(aug_data1)
            features2, seq_len2 = self._process_time_series(aug_data2)
            return {
                'features1': features1, 'features2': features2,
                'seq_len1': seq_len1, 'seq_len2': seq_len2,
                'index': index
            }

        elif self.mode == 'all':
            patient_df = self.train_data[index]
            target = self.noise_label[index]
            soft_target = target
            features, seq_len = self._process_time_series(patient_df)
            return {
                'features': features, 'seq_len': seq_len,
                'target': soft_target, 'index': index
            }

        elif self.mode == 'eval':
            patient_df = self.es_data[index]
            target = self.es_label[index]
            soft_target = target
            features, seq_len = self._process_time_series(patient_df)
            return {
                'features': features, 'seq_len': seq_len,
                'target': soft_target, 'index': index
            }

        elif self.mode == 'test':
            patient_df = self.test_data[index]
            target = self.test_label[index]
            features, seq_len = self._process_time_series(patient_df)
            return {
                'features': features, 'seq_len': seq_len,
                'target': target, 'index': index
            }

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        elif self.mode == 'eval':
            return len(self.es_data)
        else:
            return len(self.train_data)


class nct_dataloader:
    def __init__(self, args, num_workers, root_dir, log, train_zip, es_zip, test_zip,
                 noise_file='', use_soft_labels=False, soft_label_method=None, soft_label_config=None):
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        self.use_soft_labels = use_soft_labels
        self.soft_label_method = soft_label_method
        self.soft_label_config = soft_label_config

        train_zip, es_zip, test_zip = process_data(train_zip, es_zip, test_zip)
        self.train_zip = train_zip
        self.es_zip = es_zip
        self.test_zip = test_zip

    def run(self, mode, pred=None, prob=None):
        if mode == 'warmup':
            all_dataset = nct_dataset(
                mode="all", train_zip=self.train_zip, es_zip=self.es_zip, test_zip=self.test_zip,
                use_soft_labels=self.use_soft_labels, soft_label_method=self.soft_label_method,
                soft_label_config=self.soft_label_config)
            trainloader = DataLoader(
                dataset=all_dataset, batch_size=self.batch_size * 2,
                shuffle=True, num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            pred = np.array(pred) if not isinstance(pred, np.ndarray) else pred
            total_samples = len(self.train_zip[0])
            labeled_count = np.sum(pred)
            unlabeled_count = total_samples - labeled_count
            print(f"Total samples: {total_samples}, Labeled: {labeled_count}, Unlabeled: {unlabeled_count}")

            if self.dataset == 'NCT03041311':
                min_samples = max(1, self.batch_size // 16)
            else:
                min_samples = max(1, self.batch_size // 4)

            if labeled_count < min_samples:
                print(f"Warning: Too few labeled samples ({labeled_count}), adjusting...")
                indices = np.arange(total_samples)
                np.random.shuffle(indices)
                pred = np.zeros(total_samples, dtype=bool)
                pred[indices[:min_samples]] = True
                labeled_count = min_samples
                unlabeled_count = total_samples - labeled_count

            if unlabeled_count < min_samples:
                print(f"Warning: Too few unlabeled samples ({unlabeled_count}), adjusting...")
                labeled_indices = np.where(pred)[0]
                if len(labeled_indices) > total_samples - min_samples:
                    remove_count = len(labeled_indices) - (total_samples - min_samples)
                    remove_indices = np.random.choice(labeled_indices, remove_count, replace=False)
                    pred[remove_indices] = False
                    labeled_count = np.sum(pred)
                    unlabeled_count = total_samples - labeled_count

            print(f"Adjusted - Labeled: {labeled_count}, Unlabeled: {unlabeled_count}")

            labeled_dataset = nct_dataset(
                mode="labeled", pred=pred, probability=prob, log=self.log,
                train_zip=self.train_zip, es_zip=self.es_zip, test_zip=self.test_zip,
                use_soft_labels=self.use_soft_labels, soft_label_method=self.soft_label_method,
                soft_label_config=self.soft_label_config)

            unlabeled_dataset = nct_dataset(
                mode="unlabeled", pred=pred,
                train_zip=self.train_zip, es_zip=self.es_zip, test_zip=self.test_zip,
                use_soft_labels=self.use_soft_labels, soft_label_method=self.soft_label_method,
                soft_label_config=self.soft_label_config)

            if len(labeled_dataset) == 0:
                raise ValueError(f"Labeled dataset is empty! pred sum: {np.sum(pred)}")
            if len(unlabeled_dataset) == 0:
                raise ValueError(f"Unlabeled dataset is empty! pred sum: {np.sum(pred)}")

            labeled_batch_size = min(self.batch_size, len(labeled_dataset))
            unlabeled_batch_size = min(self.batch_size, len(unlabeled_dataset))

            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, batch_size=labeled_batch_size,
                shuffle=True, num_workers=self.num_workers, drop_last=True)

            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, batch_size=unlabeled_batch_size,
                shuffle=True, num_workers=self.num_workers, drop_last=True)

            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'eval':
            eval_dataset = nct_dataset(
                mode='eval', train_zip=self.train_zip, es_zip=self.es_zip, test_zip=self.test_zip,
                use_soft_labels=self.use_soft_labels, soft_label_method=self.soft_label_method,
                soft_label_config=self.soft_label_config)
            eval_loader = DataLoader(
                dataset=eval_dataset, batch_size=self.batch_size // 4,
                shuffle=False, num_workers=self.num_workers)
            return eval_loader

        elif mode == 'test':
            test_dataset = nct_dataset(
                mode='test', train_zip=self.train_zip, es_zip=self.es_zip, test_zip=self.test_zip,
                use_soft_labels=False, soft_label_method=self.soft_label_method,
                soft_label_config=self.soft_label_config)
            test_loader = DataLoader(
                dataset=test_dataset, batch_size=self.batch_size // 4,
                shuffle=False, num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            et_dataset = nct_dataset(
                mode='all', train_zip=self.train_zip, es_zip=self.es_zip, test_zip=self.test_zip,
                use_soft_labels=self.use_soft_labels, soft_label_method=self.soft_label_method,
                soft_label_config=self.soft_label_config)
            et_loader = DataLoader(
                dataset=et_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers)
            return et_loader
