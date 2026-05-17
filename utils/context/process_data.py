"""
TrialAgents 数据处理模块

从 m-agents/utils/process_data.py 迁移至此
提供临床试验数据集的加载和处理功能

依赖:
    sas7bdat: 用于读取SAS格式的临床试验数据（懒加载）
    pandas, numpy, torch, tqdm
"""

import os
import pickle
import copy
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# sas7bdat 懒加载
def _get_sas7bdat():
    try:
        from sas7bdat import SAS7BDAT
        return SAS7BDAT
    except ImportError:
        raise ImportError(
            "sas7bdat is required for loading clinical trial data. "
            "Install it with: pip install sas7bdat"
        )


def generate(type, cycle_width, start_time, end_time, max_length, coeffs=None, drop_thresh=-1.5, seed=0, random_std=1):
    """
    生成合成时间序列数据

    Args:
        type: 'indicator' 或 'linear' 或 'sin'
        cycle_width: 周期宽度
        start_time: 开始时间
        end_time: 结束时间
        max_length: 最大长度
        coeffs: 系数（用于linear和sin类型）
        drop_thresh: 丢弃阈值
        seed: 随机种子
        random_std: 随机标准差

    Returns:
        包含occurrence和values的字典
    """
    np.random.seed(seed)
    start_time = np.clip(int(start_time + np.random.normal(loc=0, scale=random_std)), 0, max_length - 1)
    end_time = np.clip(int(end_time + np.random.normal(loc=0, scale=random_std)), start_time + 1, max_length)

    if type == 'indicator':
        occurrence = [start_time]
        values = [0]
    else:
        coeffs = [coeff + np.random.normal(loc=0, scale=random_std) for coeff in coeffs]

        occurrence = np.arange(start_time, end_time, cycle_width)
        noise = np.random.normal(loc=0, scale=random_std, size=len(occurrence))
        occurrence = [occurrence[i] for i in range(len(occurrence)) if noise[i] >= drop_thresh]
        if len(occurrence) == 0:
            occurrence = [start_time]

        if type == 'linear':
            slope, intercept = coeffs
            values = [occurrence_ * slope + intercept for occurrence_ in occurrence]
        elif type == 'sin':
            amplitude, period, phase = coeffs
            values = [amplitude * np.sin(occurrence_ * 2 * np.pi / period + phase) for occurrence_ in occurrence]
        else:
            raise ValueError('Invalid type: {}'.format(type))

    return {"occurrence": occurrence, "values": values}


def load_synthetic(seed=0):
    """
    加载合成数据集

    Args:
        seed: 随机种子

    Returns:
        患者数据列表
    """
    np.random.seed(seed)

    num_events_to_generate = 850
    num_patients = 5500
    avg_num_events_per_patient = 15
    max_length = 2000

    # 生成事件配置
    event_configs = []
    for i in range(num_events_to_generate):
        type = np.random.choice(['linear', 'indicator'])
        cycle_width = np.random.randint(5, 25)
        start_time = np.random.randint(1, 20)
        end_time = np.random.randint(start_time, max_length)
        if type == 'indicator':
            coeffs = None
            start_time = np.random.randint(1, end_time)
        elif type == 'linear':
            slope = np.random.randint(-10, 10)
            intercept = np.random.randint(-10, 10)
            coeffs = [slope, intercept]

        event_configs.append({
            'type': type, 'cycle_width': cycle_width, 'start_time': start_time,
            'end_time': end_time, 'max_length': max_length, 'coeffs': coeffs
        })

    # 生成患者数据
    patients_list = []
    for patient_ind in range(num_patients):
        events_to_sample = max(int(np.random.normal(loc=avg_num_events_per_patient, scale=2)), 1)
        event_inds = np.sort(np.random.choice(len(event_configs), events_to_sample, replace=False))

        event_masks = np.zeros(len(event_configs), dtype=bool)
        event_masks[event_inds] = True

        occurrence_padded = np.zeros((len(event_inds), max_length), dtype=bool)
        values_padded = np.zeros((len(event_inds), max_length))

        for i, event_ind in enumerate(event_inds):
            ret = generate(**event_configs[event_ind], seed=patient_ind, random_std=1)
            occurrence_padded[i, ret['occurrence']] = 1
            values_padded[i, ret['occurrence']] = ret['values']

        patients_list.append({
            'event_masks': event_masks,
            'occurrence': occurrence_padded,
            'values': values_padded
        })

    # 统计每个事件的发生次数
    all_event_occurrences = {i: [] for i in range(num_events_to_generate)}
    for patient in patients_list:
        for i, event_ind in enumerate(patient['event_masks'].nonzero()[0]):
            all_event_occurrences[event_ind].append(patient['occurrence'][i])

    all_event_occurrences = {i: np.array(all_event_occurrences[i]) for i in range(num_events_to_generate)}

    # 标准化事件值
    all_event_means = {}
    all_event_stds = {}
    all_event_values = {i: [] for i in range(num_events_to_generate)}

    for patient in patients_list:
        for i, event_ind in enumerate(patient['event_masks'].nonzero()[0]):
            all_event_values[event_ind].append(patient['values'][i][patient['occurrence'][i].astype(bool)])

    all_event_values = {i: np.concatenate(all_event_values[i]) for i in all_event_values.keys()}
    all_event_means = {i: np.nanmean(all_event_values[i]) for i in all_event_values.keys()}
    all_event_stds = {i: np.nanstd(all_event_values[i]) for i in all_event_stds.keys()}

    for patient in patients_list:
        for i, event_ind in enumerate(patient['event_masks'].nonzero()[0]):
            if all_event_stds[event_ind] != 0.0:
                patient['values'][i] = (patient['values'][i] - all_event_means[event_ind]) / all_event_stds[event_ind]
            else:
                patient['values'][i] = patient['values'][i] - all_event_means[event_ind]
            patient['values'][i] = patient['values'][i] * patient['occurrence'][i]

    for patient in patients_list:
        patient['event_masks'] = torch.Tensor(patient['event_masks']).long()
        patient['occurrence'] = torch.Tensor(patient['occurrence']).long()
        patient['values'] = torch.Tensor(patient['values']).float()

    return patients_list


def convert_dict_to_df(patient_list, indicator_col, value_col, patient_id_col, time_col, label_col):
    """
    将患者列表转换为DataFrame

    Args:
        patient_list: 患者数据列表
        indicator_col: 指标列名
        value_col: 值列名
        patient_id_col: 患者ID列名
        time_col: 时间列名
        label_col: 标签列名

    Returns:
        pd.DataFrame
    """
    df_dict = {
        patient_id_col: [],
        indicator_col: [],
        value_col: [],
        time_col: [],
        label_col: []
    }

    for pat_i, patient in enumerate(patient_list):
        patient_events, patient_values, patient_times, label = patient
        df_dict[patient_id_col].extend([pat_i] * len(patient_events))
        df_dict[indicator_col].extend(patient_events)
        df_dict[value_col].extend(patient_values)
        df_dict[time_col].extend(patient_times)
        df_dict[label_col].extend([label] * len(patient_events))

    reconstructed_df = pd.DataFrame(df_dict, columns=[
        patient_id_col, indicator_col, value_col, time_col, label_col
    ])
    reconstructed_df[indicator_col] = reconstructed_df[indicator_col].astype(str)
    reconstructed_df[value_col] = reconstructed_df[value_col].astype(float)
    reconstructed_df[time_col] = reconstructed_df[time_col].astype(float)
    reconstructed_df[label_col] = reconstructed_df[label_col].astype(int)
    reconstructed_df = reconstructed_df.sort_values(by=[patient_id_col, time_col]).reset_index(drop=True)
    reconstructed_df[patient_id_col] = reconstructed_df[patient_id_col].astype(str)
    return reconstructed_df


def process_patient_df(df, patient_id_col, indicator_col, value_col, time_col, label_col, demographic_cols=None):
    """
    处理患者DataFrame，将数据转换为列表格式

    Args:
        df: 原始DataFrame
        patient_id_col: 患者ID列
        indicator_col: 指标列
        value_col: 值列
        time_col: 时间列（可以是单列或列表）
        label_col: 标签列
        demographic_cols: 人口统计学列列表

    Returns:
        患者数据列表
    """
    if len(time_col) == 1:
        patient_list = df.groupby(patient_id_col).apply(
            lambda x: [
                x[patient_id_col].values[0],  # patient_id
                [x[col].values[0] for col in demographic_cols] if demographic_cols else None,  # demographic data
                x[indicator_col].tolist(),  # event indicators
                x[value_col].tolist(),  # event values
                x[time_col[0]].tolist(),  # event times
                x[label_col].values[0],  # label
            ]
        ).tolist()
    else:
        patient_list = df.groupby(patient_id_col).apply(
            lambda x: [
                x[patient_id_col].values[0],  # patient_id
                [x[col].values[0] for col in demographic_cols] if demographic_cols else None,  # demographic data
                x[indicator_col].tolist(),  # event indicators
                x[value_col].tolist(),  # event values
                x[time_col[0]].tolist(),  # event start times
                x[time_col[1]].tolist(),  # event end times
                x[label_col].values[0],  # label
            ]
        ).tolist()

    return patient_list


def process_patient_df_as_dict(df, patient_id_col, indicator_col, value_col, time_col, label_col, demographic_cols=None):
    """
    将患者数据处理为字典格式

    Args:
        df: DataFrame
        patient_id_col: 患者ID列
        indicator_col: 指标列
        value_col: 值列
        time_col: 时间列
        label_col: 标签列
        demographic_cols: 人口统计学列

    Returns:
        患者ID到数据的字典
    """
    patient_dict = {}

    for patient_id, group in df.groupby(patient_id_col):
        patient_data = {
            'events': group[indicator_col].tolist(),
            'values': group[value_col].tolist(),
            'times': group[time_col].tolist(),
            'label': group[label_col].values[0]
        }

        if demographic_cols:
            for i, col in enumerate(demographic_cols):
                patient_data[col] = group[col].values[0]

        patient_dict[patient_id] = patient_data

    return patient_dict


def clean_bytes_string(val):
    """
    清理字节字符串格式

    Args:
        val: 原始值

    Returns:
        清理后的字符串
    """
    if isinstance(val, bytes):
        return val.decode('utf-8')
    elif isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
        return val[2:-1]
    else:
        return str(val)


def load_clinical_trial_datasets(trial_path, dataset_to_process=None, task=None, check_reconstruction=False, min_events_per_patient=2):
    """
    加载临床试验数据集

    Args:
        trial_path: 临床试验数据根目录
        dataset_to_process: 要处理的数据集名称列表，如 ['NCT00003299', 'NCT00694382']
        task: 任务类型
        check_reconstruction: 是否检查数据重构正确性
        min_events_per_patient: 每个患者最少事件数

    Returns:
        处理后的数据参数字典列表
    """
    if dataset_to_process is None:
        dataset_to_process = [
            'NCT00003299', 'NCT00041119', 'NCT00079274',
            'NCT00174655', 'NCT00312208', 'NCT00694382', 'NCT03041311'
        ]

    args = []

    # ==================== NCT00003299 (SCLC, 561 subj) ====================
    if 'NCT00003299' in dataset_to_process:
        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00003299/c9732_ae.sas7bdat'), skip_header=False) as reader:
            df = reader.to_data_frame()
        df['time'] = df.groupby('PHATOM_ID').cumcount() + 1
        df = df[['PHATOM_ID', 'AE_NAME', 'AE_GRADE', 'time']]

        # labels
        df_label = pd.read_csv(os.path.join(trial_path, 'NCT00003299/NCT00003299.csv'), index_col=0)
        df_label = df_label.rename(columns={'patient_id': 'PHATOM_ID', 'target_label': 'label'})
        df = df.merge(df_label[['PHATOM_ID', 'label']], on='PHATOM_ID', how='left')
        df['label'] = df['label'].replace({0: 1, 1: 0})

        # demographic info
        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00003299/c9732_demographic.sas7bdat'), skip_header=False) as reader:
            df_demo = reader.to_data_frame()
        df = df.merge(df_demo[['PHATOM_ID', 'AGE', 'GENDER', 'PS', 'WGT_LOSSPCT', 'NUM_META', 'SCLC_STAGE', 'RADIOTHERAPY', 'CHEMO_CYCLE']], on='PHATOM_ID', how='left')

        args.append({
            'df': df,
            'indicator_col': 'AE_NAME',
            'value_col': 'AE_GRADE',
            'patient_id_col': 'PHATOM_ID',
            'time_col': ['time'],
            'label_col': 'label',
            'demographic_cols': ['AGE', 'GENDER', 'PS', 'WGT_LOSSPCT', 'NUM_META', 'SCLC_STAGE', 'RADIOTHERAPY', 'CHEMO_CYCLE']
        })

    # ==================== NCT00041119 ====================
    if 'NCT00041119' in dataset_to_process:
        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00041119/aeclean_finalb.sas7bdat'), skip_header=False) as reader:
            df = reader.to_data_frame()
        df['time'] = df.groupby('MASK_ID').cumcount() + 1
        df = df[['MASK_ID', 'eventname', 'eventcat', 'GRADE_ID', 'time']]
        df['eventname'] = df['eventname'] + ' (' + df['eventcat'] + ')'

        df_label = pd.read_csv(os.path.join(trial_path, 'NCT00041119/NCT00041119.csv'), index_col=0)
        df_label = df_label.rename(columns={'mask_id': 'MASK_ID', 'target_label': 'label'})
        df = df.merge(df_label[['MASK_ID', 'label']], on='MASK_ID', how='left')

        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00041119/all_finalb.sas7bdat'), skip_header=False) as reader:
            df_demo = reader.to_data_frame()
        df_demo = df_demo.rename(columns={'mask_id': 'MASK_ID'})
        df = df.merge(df_demo[['MASK_ID', 'ETHNIC_ID', 'GROUP_ID', 'RACE_ID',
            'stra1', 'stra2', 'stra3', 'elig', 'OH002',
            'OH003', 'OH004', 'OH005', 'OH011', 'OH012', 'OH013', 'OH014', 'OH016',
            'OH027', 'OH028', 'OH032', 'OH036', 'OH037', 'tsize', 'agecat',
            'num_pos_nodes']], on='MASK_ID', how='left')

        args.append({
            'df': df,
            'patient_id_col': 'MASK_ID',
            'demographic_cols': ['ETHNIC_ID', 'GROUP_ID', 'RACE_ID', 'stra1', 'stra2', 'stra3', 'elig', 'OH002',
                                 'OH003', 'OH004', 'OH005', 'OH011', 'OH012', 'OH013', 'OH014', 'OH016',
                                 'OH027', 'OH028', 'OH032', 'OH036', 'OH037', 'tsize', 'agecat', 'num_pos_nodes'],
            'indicator_col': 'eventname',
            'value_col': 'GRADE_ID',
            'time_col': ['time'],
            'label_col': 'label'
        })

    # ==================== NCT00079274 ====================
    if 'NCT00079274' in dataset_to_process:
        df = pd.read_csv(os.path.join(trial_path, 'NCT00079274/tox.csv'))
        df['time'] = df.groupby('mask_id').cumcount() + 1
        df = df[['mask_id', 'tox', 'GRADE', 'time']]

        df_obj = pd.read_csv(os.path.join(trial_path, 'NCT00079274/objectives.csv'))
        df_obj['label'] = df_obj['fustat8']
        df = df.merge(df_obj[['mask_id', 'label']], on='mask_id', how='left')

        df_demo = pd.read_csv(os.path.join(trial_path, 'NCT00079274/NCT00079274.csv'))
        df_demo = df_demo[['mask_id', 'age', 'sex', 'race', 'bmi', 'ECOG performance score', 'histology', 'biomarker KRAS']]

        df = df.merge(df_demo, on='mask_id', how='left')

        args.append({
            'df': df,
            'patient_id_col': 'mask_id',
            'demographic_cols': ['age', 'sex', 'race', 'bmi', 'ECOG performance score', 'histology', 'biomarker KRAS'],
            'indicator_col': 'tox',
            'value_col': 'GRADE',
            'time_col': ['time'],
            'label_col': 'label'
        })

    # ==================== NCT00174655 ====================
    if 'NCT00174655' in dataset_to_process:
        df_demo = pd.read_csv(os.path.join(trial_path, 'NCT00174655/data', 'demog.csv'))
        df_demo = df_demo[['RUSUBJID', 'AGE', 'SEX', 'RACE', 'CBPYN', 'ZVISNAME', 'ZVISDATE_DY']]

        df_diag = pd.read_csv(os.path.join(trial_path, 'NCT00174655/data', 'diag2.csv'))
        df_diag = df_diag[['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY', 'HISGRADE', 'HISTYPE', 'MARGIN', 'MULTUYN', 'PATHOM', 'PATHON', 'PATHOT', 'POSALN', 'PTSIZE', 'RESALN', 'SITE_L']]

        df_meno = pd.read_csv(os.path.join(trial_path, 'NCT00174655/data', 'meno.csv'))
        df_meno = df_meno[['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY', 'BILOVARI', 'HORMOREP', 'HYSTEREC']]

        df_label = pd.read_csv(os.path.join(trial_path, 'NCT00174655/data/death.csv'))
        df_demo['label'] = (df_demo['RUSUBJID'].isin(df_label['RUSUBJID'])).astype(float)

        df_basic = df_demo.merge(df_diag, on=['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY'], how='left')
        df_basic = df_basic.merge(df_meno, on=['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY'], how='left')
        for col in ['SEX', 'RACE']:
            df_basic[col] = df_basic[col].apply(clean_bytes_string)

        df_ae = pd.read_csv(os.path.join(trial_path, 'NCT00174655/data/ae.csv'))
        df_ae = df_ae[['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY', 'AECOSE', 'GRAD', 'AEREL', 'AESER', 'AESTAT']]
        df_ae = df_ae.dropna(subset=['AECOSE', 'GRAD'], how='all')
        df_ae = df_ae.merge(df_basic, on=['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY'], how='left')
        df_ae['label'] = df_ae['RUSUBJID'].isin(df_label['RUSUBJID']).astype(float)
        for col in ['AECOSE', 'ZVISNAME']:
            df_ae[col] = df_ae[col].apply(clean_bytes_string)

        df_vital = pd.read_csv(os.path.join(trial_path, 'NCT00174655/data/vital.csv'))
        df_vital = df_vital[['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY', 'D_VITAL_DY', 'PSKAR', 'WT', 'HT']]
        df_vital = df_vital.dropna(subset=['PSKAR', 'WT', 'HT'], how='all')
        df_vital['HT'] = df_vital.groupby(['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY'])['HT'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        df_vital['label'] = df_vital['RUSUBJID'].isin(df_label['RUSUBJID']).astype(float)
        df_vital = df_vital.merge(df_basic, on=['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY'], how='left')
        df_vital['label'] = df_vital['RUSUBJID'].isin(df_label['RUSUBJID']).astype(float)
        for col in ['PSKAR', 'WT', 'HT', 'ZVISNAME']:
            df_vital[col] = df_vital[col].apply(clean_bytes_string)
        df_vital['VS'] = '[ Performance Status:' + df_vital['PSKAR'].astype(str) + ', Weight: ' + df_vital['WT'].astype(str) + 'kg, Height: ' + df_vital['HT'].astype(str) + 'cm. ]'

        df_ekg = pd.read_csv(os.path.join(trial_path, 'NCT00174655/data/ekg.csv'))
        df_ekg = df_ekg[['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY', 'D_ASSESS_DY', 'EINTNA']]
        df_ekg = df_ekg.dropna(subset=['EINTNA'], how='all')
        df_ekg = df_ekg.merge(df_basic, on=['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY'], how='left')
        df_ekg['label'] = df_ekg['RUSUBJID'].isin(df_label['RUSUBJID']).astype(float)
        for col in ['EINTNA', 'ZVISNAME']:
            df_ekg[col] = df_ekg[col].apply(clean_bytes_string)

        df_labs = pd.read_csv(os.path.join(trial_path, 'NCT00174655/data/lab.csv'))
        df_labs = df_labs[['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY', 'D_LABSAM_DY', 'LTEST', 'SLVALUE', 'RECUNIT', 'SUPVALUE']]
        df_labs = df_labs.dropna(subset=['LTEST', 'SLVALUE', 'RECUNIT', 'SUPVALUE'], how='all')
        df_labs = df_labs.merge(df_basic, on=['RUSUBJID', 'ZVISNAME', 'ZVISDATE_DY'], how='left')
        df_labs['label'] = df_labs['RUSUBJID'].isin(df_label['RUSUBJID']).astype(float)
        for col in ['LTEST', 'RECUNIT', 'ZVISNAME']:
            df_labs[col] = df_labs[col].apply(clean_bytes_string)
        df_labs['LABS'] = '[' + df_labs['LTEST'].astype(str) + ', with converted value: ' + df_labs['SLVALUE'].astype(str) + ' ' + df_labs['RECUNIT'].astype(str) + ' (Converted Upper Normal Value: ' + df_labs['SUPVALUE'].astype(str) + '). ]'

        for df_name in ['ae', 'vital', 'ekg', 'labs']:
            args.append({
                'df': eval(f'df_{df_name}'),
                'patient_id_col': 'RUSUBJID',
                'demographic_cols': ['AGE', 'SEX', 'RACE', 'CBPYN', 'HISGRADE', 'HISTYPE', 'MARGIN', 'MULTUYN', 'PATHOM', 'PATHON', 'PATHOT', 'POSALN', 'PTSIZE', 'RESALN', 'SITE_L', 'BILOVARI', 'HORMOREP', 'HYSTEREC'],
                'indicator_col': 'AECOSE' if df_name == 'ae' else 'VS' if df_name == 'vital' else 'EINTNA' if df_name == 'ekg' else 'LABS',
                'value_col': 'GRAD' if df_name == 'ae' else 'VS' if df_name == 'vital' else 'EINTNA' if df_name == 'ekg' else 'LABS',
                'time_col': ['ZVISDATE_DY', 'ZVISNAME'],
                'label_col': 'label',
            })

    # ==================== NCT00312208 ====================
    if 'NCT00312208' in dataset_to_process:
        df = pd.read_csv(os.path.join(trial_path, 'NCT00312208/data/ae.csv'))
        df = df[['RUSUBJID', 'AECOS', 'GRAD', 'STRTDATE_DY']]
        df['RUSUBJID'] = df['RUSUBJID'].str.slice(2, -1)

        df_label = pd.read_csv(os.path.join(trial_path, 'NCT00312208/NCT00312208.csv'), index_col=0)
        df_label = df_label.rename(columns={'target_label': 'label'})
        df = df.merge(df_label[['RUSUBJID', 'label']], on='RUSUBJID', how='left')

        args.append({
            'df': df,
            'indicator_col': 'AECOS',
            'value_col': 'GRAD',
            'patient_id_col': 'RUSUBJID',
            'time_col': 'STRTDATE_DY',
            'label_col': 'label'
        })

    # ==================== NCT00694382 (VTE) ====================
    if 'NCT00694382' in dataset_to_process:
        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00694382/dm.sas7bdat'), skip_header=False) as reader:
            df_dm = reader.to_data_frame()
        df_dm = df_dm[['RUSUBJID', 'AGEC', 'SEX', 'RACE', 'REGION', 'RFSTDY', 'RFENDY', 'COMPLT']]

        df_label = pd.read_csv(os.path.join(trial_path, 'NCT00694382/NCT00694382.csv'), index_col=0)
        df_label = df_label.rename(columns={'target_label': 'label'})

        df_basic = df_dm.merge(df_label[['RUSUBJID', 'label']], on='RUSUBJID', how='left')

        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00694382/ae.sas7bdat'), skip_header=False) as reader:
            df_ae = reader.to_data_frame()
        df_ae['AESEV'] = df_ae['AESEV'].replace({'MILD': 1, 'MODERATE': 2, 'SEVERE': 3, '': np.nan}).astype(float)
        df_ae = df_ae[['RUSUBJID', 'AEDECOD', 'AESEV', 'AESTDY', 'AEENDY']]
        df_ae = df_basic.merge(df_ae, on='RUSUBJID', how='left')

        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00694382/ex.sas7bdat'), skip_header=False) as reader:
            df_ex = reader.to_data_frame()
        df_ex['EXROUTE'] = df_ex['EXROUTE'] + ' ' + df_ex['EXDOSFRM']
        df_ex = df_ex[['RUSUBJID', 'EXDOSE', 'EXROUTE', 'EXSTDY', 'EXENDY']]
        df_ex = df_basic.merge(df_ex, on='RUSUBJID', how='left')

        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00694382/cm.sas7bdat'), skip_header=False) as reader:
            df_cm = reader.to_data_frame()
        df_cm = df_cm[['RUSUBJID', 'CMDECOD', 'CMDOSE', 'CMROUTE', 'CMSTDY', 'CMENDY', 'CMINDC']]
        df_cm = df_basic.merge(df_cm, on='RUSUBJID', how='left')

        with _get_sas7bdat()(os.path.join(trial_path, 'NCT00694382/vs.sas7bdat'), skip_header=False) as reader:
            df_vs = reader.to_data_frame()
        df_vs['VSTEST'] = df_vs['VSTEST'] + '(unit: ' + df_vs['VSSTRESU'] + ')'
        df_vs = df_vs[['RUSUBJID', 'VSTEST', 'VSSTRESN', 'VSSTRESU', 'VISIT', 'VSDY']]
        df_vs = df_basic.merge(df_vs, on='RUSUBJID', how='left')

        for df_name in ['ae', 'ex', 'cm', 'vs']:
            args.append({
                'df': eval(f'df_{df_name}'),
                'patient_id_col': 'RUSUBJID',
                'demographic_cols': ['AGEC', 'SEX', 'RACE', 'REGION', 'RFSTDY', 'RFENDY', 'COMPLT'],
                'indicator_col': 'AEDECOD' if df_name == 'ae' else 'EXROUTE' if df_name == 'ex' else 'CMDECOD' if df_name == 'cm' else 'VSTEST',
                'value_col': 'AESEV' if df_name == 'ae' else 'EXDOSE' if df_name == 'ex' else 'CMDOSE' if df_name == 'cm' else 'VSSTRESN',
                'time_col': ['AESTDY', 'AEENDY'] if df_name == 'ae' else ['EXSTDY', 'EXENDY'] if df_name == 'ex' else ['CMSTDY', 'CMENDY'] if df_name == 'cm' else ['VSDY'],
                'label_col': 'label'
            })

    # ==================== NCT03041311 (SCLC, 51 subj) ====================
    if 'NCT03041311' in dataset_to_process:
        with _get_sas7bdat()(os.path.join(trial_path, 'NCT03041311/adae.sas7bdat'), skip_header=False) as reader:
            df = reader.to_data_frame()
        df['AETOXGR'] = df['AETOXGR'].astype(int, errors='ignore')
        df = df[['USUBJID', 'AEDECOD', 'AETOXGR', 'ASTDY', 'AENDY']]

        df_label = pd.read_csv(os.path.join(trial_path, 'NCT03041311/NCT03041311.csv'), index_col=0)
        df_label = df_label.rename(columns={'target_label': 'label'})
        df = df.merge(df_label[['USUBJID', 'label']], on='USUBJID', how='left')

        with _get_sas7bdat()(os.path.join(trial_path, 'NCT03041311/adsl.sas7bdat'), skip_header=False) as reader:
            df_demo = reader.to_data_frame()
        df = df.merge(df_demo[['USUBJID', 'AGE', 'SEX', 'HTCM', 'WTKG', 'ECOG', 'ECOGSCR', 'SMKSTAT', 'SMKSTAGE', 'SMKENAGE']], on='USUBJID', how='left')

        args.append({
            'df': df,
            'indicator_col': 'AEDECOD',
            'value_col': 'AETOXGR',
            'patient_id_col': 'USUBJID',
            'time_col': ['ASTDY', 'AENDY'],
            'label_col': 'label',
            'demographic_cols': ['AGE', 'SEX', 'HTCM', 'WTKG', 'ECOG', 'ECOGSCR', 'SMKSTAT', 'SMKSTAGE', 'SMKENAGE']
        })

    # ==================== 处理数据集 ====================
    for i in range(len(args)):
        raw_df = args[i]['df']
        df = raw_df.copy()
        indicator_col = args[i]['indicator_col']
        value_col = args[i]['value_col']
        patient_id_col = args[i]['patient_id_col']
        time_col = args[i]['time_col']
        label_col = args[i]['label_col']

        # 处理time_col
        if isinstance(time_col, list):
            time_col = time_col[0]

        # 删除空值
        df = df.dropna(subset=[indicator_col, value_col, patient_id_col, time_col, label_col], how='any')

        # 类型转换
        df[indicator_col] = df[indicator_col].astype(str)
        if 'NCT00174655' in dataset_to_process:
            df[value_col] = df[value_col].astype(str)
        else:
            df[value_col] = df[value_col].astype(float)
        df[patient_id_col] = df[patient_id_col].astype(str)
        df[time_col] = df[time_col].astype(float)
        df[label_col] = df[label_col].astype(int)

        # 过滤事件数过少的患者
        df = df.groupby(patient_id_col).filter(lambda x: len(x) >= min_events_per_patient)

        # 统计信息
        max_occurrences_per_event = df.groupby([patient_id_col, indicator_col]).size().max()
        all_events = np.unique(df[indicator_col])
        patient_list = np.unique(df[patient_id_col])
        print(f'len subjid: {len(patient_list)}, max max_occurrences_per_event: {max_occurrences_per_event}, num events: {len(all_events)}')

        # 去重
        df = df.drop_duplicates(subset=[patient_id_col, time_col, indicator_col])
        df = df.reset_index(drop=True)
        df = df.sort_values(by=[patient_id_col, time_col]).reset_index(drop=True)
        args[i]['df'] = df.copy()

    return args


# Backward-compatible aliases for code migrated from m-agents
def process_patient_df_old(df, patient_id_col, indicator_col, value_col, time_col, label_col, demographic_col=None):
    """Alias for process_patient_df with old parameter name (demographic_col singular)."""
    return process_patient_df(df, patient_id_col, indicator_col, value_col, time_col, label_col, demographic_cols=demographic_col)


def load_clinical_trial_datasets_old(trial_path, dataset_to_process=None, check_reconstruction=False, min_events_per_patient=2):
    """Alias for load_clinical_trial_datasets without task parameter."""
    return load_clinical_trial_datasets(trial_path, dataset_to_process=dataset_to_process, check_reconstruction=check_reconstruction, min_events_per_patient=min_events_per_patient)


__all__ = [
    'generate',
    'load_synthetic',
    'convert_dict_to_df',
    'process_patient_df',
    'process_patient_df_as_dict',
    'load_clinical_trial_datasets',
    'process_patient_df_old',
    'load_clinical_trial_datasets_old',
    'clean_bytes_string',
]
