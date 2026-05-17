"""
TrialAgents 医疗上下文构建模块

从 m-agents/utils/healthcare_context_utils.py 迁移至此
提供患者医疗上下文生成功能

依赖:
    utils.context.datasets_info: 医疗数据字典
    utils.context.process_data: 数据处理函数
"""

import os
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .datasets_info import (
    medical_name,
    medical_unit,
    medical_standard,
    medical_standard_for_esrd,
    disease_english,
    original_disease,
    mimic_range,
    mimic_unit,
)


def get_data_from_files(
    data_url: str,
    task: str,
    model: str,
    mode: str,
    patient_index: int = 0
) -> Tuple:
    """
    从文件加载患者数据

    Args:
        data_url: 数据目录
        task: 任务名称
        model: 模型名称
        mode: 模式 (test/val)
        patient_index: 患者索引

    Returns:
        (x, raw_x, record_time, features, lab_features, note, total_y, y, important_features)
    """
    y = None
    important_features = None
    total_y = None

    x = pd.read_pickle(os.path.join(data_url, f"{mode}_x.pkl"))[patient_index]
    raw_x = pd.read_pickle(os.path.join(data_url, f"{mode}_raw_x.pkl"))[patient_index]

    if 'mimic-iv' in data_url and task == 'readmission':
        if os.path.exists(os.path.join(data_url, "dl_data", f"{model}_readmission_{mode}_output.pkl")):
            total_y = pd.read_pickle(os.path.join(data_url, "dl_data", f"{model}_readmission_{mode}_output.pkl"))
            y = total_y[patient_index]
            important_features = pd.read_pickle(os.path.join(data_url, "dl_data", f"{model}_readmission_{mode}_features.pkl"))[patient_index]
    elif os.path.exists(os.path.join(data_url, "dl_data", f"{model}_outcome_{mode}_output.pkl")):
        total_y = pd.read_pickle(os.path.join(data_url, "dl_data", f"{model}_outcome_{mode}_output.pkl"))
        y = total_y[patient_index]
        important_features = pd.read_pickle(os.path.join(data_url, "dl_data", f"{model}_outcome_{mode}_features.pkl"))[patient_index]
    else:
        print("Output file not found.")

    if 'mimic-iv' in data_url:
        features = pd.read_pickle(os.path.join(data_url, "numerical_features.pkl"))
        lab_features = pd.read_pickle(os.path.join(data_url, "labtest_features.pkl"))
        if mode == 'test':
            note = pd.read_pickle(os.path.join(data_url, "test_notes.pkl"))[patient_index]
        elif mode == 'val':
            note = pd.read_pickle(os.path.join(data_url, "val_notes.pkl"))[patient_index]
    else:
        features = pd.read_pickle(os.path.join(data_url, "labtest_features.pkl"))
        lab_features = None
        note = None

    if os.path.exists(os.path.join(data_url, f"{mode}_record_time.pkl")):
        record_time = pd.read_pickle(os.path.join(data_url, f"{mode}_record_time.pkl"))[patient_index]
    else:
        record_time = None

    return x, raw_x, record_time, features, lab_features, note, total_y, y, important_features


def get_var_desc(var: float) -> str:
    """获取变量描述（百分比形式）"""
    if var > 0:
        return round(var * 100, 2)
    else:
        return round(-var * 100, 2)


def get_trend_desc(var: float) -> str:
    """获取趋势描述"""
    if var > 0:
        return "increased"
    else:
        return "decreased"


def get_recommended_trend_desc(var: float) -> str:
    """获取建议的趋势描述"""
    if var > 0:
        return "decrease"
    else:
        return "increase"


def get_range_desc(key: str, var: float) -> str:
    """获取范围描述"""
    if key in ["Weight", "Appetite"]:
        return ""
    if key not in medical_standard:
        return ""
    if var < medical_standard[key][0]:
        return f"the value is lower than normal range by {round((medical_standard[key][0] - var) / medical_standard[key][0] * 100, 2)}%"
    elif var > medical_standard[key][1]:
        return f"the value is higher than normal range by {round((var - medical_standard[key][1]) / medical_standard[key][1] * 100, 2)}%"
    else:
        return "the value is within the normal range"


def get_mean_desc(var: str, mean: float) -> str:
    """获取相对于平均值的描述"""
    if var < mean:
        return f"{round((mean - var) / mean * 100, 0)}% lower"
    elif var > mean:
        return f"{round((var - mean) / mean * 100, 0)}% higher"


def get_death_desc(risk: float) -> str:
    """获取死亡风险等级描述"""
    if risk < 0.5:
        return "a low level"
    elif risk < 0.7:
        return "a high level"
    else:
        return "an extremely high level"


def get_distribution(data, values) -> str:
    """获取数据分布描述"""
    arr = np.sort(np.array(values))
    index = np.searchsorted(arr, data, side='right')
    rank = index / len(arr) * 100
    if rank < 40:
        return "at the bottom 40% levels"
    elif rank < 70:
        return "at the middle 30% levels"
    else:
        return "at the top 30% levels"


def format_input_ehr(
    raw_x: np.ndarray,
    features: List[str],
    lab_features: List[str] = None,
    record_time: List[List[str]] = None
) -> str:
    """
    格式化EHR数据为文本描述

    Args:
        raw_x: 原始特征数据
        features: 特征名称列表
        lab_features: 实验室特征列表
        record_time: 记录时间列表

    Returns:
        格式化的EHR描述文本
    """
    ehr = ""

    if lab_features is not None:  # mimic-iv
        categorical_features = [
            'Capillary refill rate', 'Glascow coma scale eye opening',
            'Glascow coma scale motor response', 'Glascow coma scale total',
            'Glascow coma scale verbal response'
        ]
        for categorical_feature in categorical_features:
            indexes = [i for i, f in enumerate(lab_features) if f.startswith(categorical_feature)]
            feature_values = []
            for visit in raw_x:
                values = [visit[i] for i in indexes]
                if 1 not in values:
                    feature_values.append('unknown')
                else:
                    for i in indexes:
                        if visit[i] == 1:
                            feature_values.append(lab_features[i].split('->')[-1])
                            break
            ehr += f"- {categorical_feature}: '" + "', ".join(feature_values) + "'\n"
        raw_x = raw_x[:, len(lab_features) - len(features):]

    for i, feature in enumerate(features):
        name = medical_name.get(feature, feature)
        ehr += f"- {name}: '" + "', ".join(list(map(lambda x: str(round(x, 2)), raw_x[:, i]))) + "'. "
        ehr += f"Unit: {medical_unit[feature]}. " if feature in medical_unit else ""
        if feature in medical_standard:
            ehr += f"Reference range for healthy people: {medical_standard[feature][0]} {medical_unit.get(feature, '')} to {medical_standard[feature][1]} {medical_unit.get(feature, '')}. "
        if feature in medical_standard_for_esrd:
            ehr += f"Reference range for ESRD patients: {medical_standard_for_esrd[feature]}.\n"
        else:
            ehr += "\n"

    if record_time is not None:
        assert len(raw_x) == len(record_time), "The length of raw_x and record_time should be the same."
        ehr += "The patient's EHR data is recorded at the following time points:\n"
        ehr += ", ".join(record_time) + ".\n"

    return ehr


def generate_prompt(
    dataset: str,
    data_url: str,
    patient_index: int,
    patient_id: int
) -> Tuple[str, str]:
    """
    生成医疗上下文prompt

    Args:
        dataset: 数据集名称
        data_url: 数据目录
        patient_index: 患者索引
        patient_id: 患者ID

    Returns:
        (subcontext, hcontext)
    """
    if dataset == 'esrd':
        basic_data = pd.read_pickle(os.path.join(data_url, 'basic.pkl'))[patient_id]
        gender = "male" if basic_data["Gender"] == 1 else "female"
        age = basic_data["Age"]
        if " " in basic_data["Origin_disease"]:
            ori_disease = basic_data["Origin_disease"].split(" ")[0]
            ori_disease = original_disease.get(ori_disease, ori_disease)
        else:
            ori_disease = original_disease.get(basic_data["Origin_disease"], basic_data["Origin_disease"])
        basic_disease = [disease_english[key] for key in disease_english.keys() if basic_data.get(key, 0) == 1]
        basic_disease = ", and basic disease " + ", ".join(basic_disease) if len(basic_disease) > 0 else ""
        basic_context = f"This {gender} patient, aged {age}, is an End-Stage Renal Disease(ESRD) patient with original disease {ori_disease}{basic_disease}.\n"
    elif dataset == 'cdsl':
        basic_data = pd.read_pickle(os.path.join(data_url, 'basic.pkl'))[patient_id]
        gender = "male" if basic_data["Sex"] == 1 else "female"
        age = basic_data["Age"]
        basic_context = f"This {gender} patient, aged {age}, is an patient admitted with a diagnosis of COVID-19 or suspected COVID-19 infection.\n"
    else:
        basic_context = '\n'

    models = ['LR', 'ConCare']
    last_visit_context = f"We have {len(models)} models {', '.join(models)} to predict the mortality risk and estimate the feature importance weight for the patient in the last visit:\n"
    for model in models:
        _, raw_x, features, y, important_features = get_data_from_files(data_url, model, patient_index)
        ehr_context = "Here is multivariate time-series electronic health record data of the patient, a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as sequences of numerical values for each feature.\n" + format_input_ehr(raw_x, features)

        last_visit = f"The mortality prediction risk for the patient from {model} model is {round(float(y), 2)} out of 1.0, which means the patient is at {get_death_desc(float(y))} of death risk. Our model especially pays great attention to the following features:\n"

        survival_stats = pd.read_pickle(os.path.join(data_url, 'survival.pkl'))
        dead_stats = pd.read_pickle(os.path.join(data_url, 'dead.pkl'))
        for item in important_features:
            key, value = item
            if key in ['Weight', 'Appetite']:
                continue
            survival_mean = survival_stats[key]['mean']
            dead_mean = dead_stats[key]['mean']
            key_name = medical_name.get(key, key)
            key_unit = ' ' + medical_unit.get(key, '') if key in medical_unit else ''
            last_visit += f'{key_name}: with '
            if model == 'ConCare':
                last_visit += f'importance weight of {round(float(value["attention"]), 3)} out of 1.0. '
            else:
                last_visit += f'shap value of {round(float(value["attention"]), 3)}. '
            last_visit += f'The feature value is {round(value["value"], 2)}{key_unit}, which is {get_mean_desc(value["value"], survival_mean)} than the average value of survival patients ({round(survival_mean, 2)}{key_unit}), {get_mean_desc(value["value"], dead_mean)} than the average value of dead patients ({round(dead_mean, 2)}{key_unit}).\n'
        last_visit_context += last_visit + '\n'

    subcontext = basic_context + last_visit_context
    hcontext = basic_context + '\n' + ehr_context + '\n' + last_visit_context

    return subcontext, hcontext


class ContextBuilder:
    """
    医疗上下文构建器

    根据患者数据生成用于LLM分析的医疗上下文
    """

    def __init__(
        self,
        dataset_name: str,
        task: str,
        model_name: str,
        mode: str
    ) -> None:
        """
        初始化上下文构建器

        Args:
            dataset_name: 数据集名称 (esrd, cdsl, mimic-iv, mimic-iii, NCT*)
            task: 任务名称
            model_name: 模型名称
            mode: 模式 (test/val)
        """
        self.dataset_name = dataset_name
        self.dataset_dir = f"ehr_datasets/{dataset_name}/processed/fold_1"
        self.task = task
        self.model_name = model_name
        self.mode = mode

    def generate_context(
        self,
        patient_index: int,
        patient_id: int,
        is_baseline: bool = False
    ):
        """
        生成患者医疗上下文

        Args:
            patient_index: 患者在数据集中的索引
            patient_id: 患者ID
            is_baseline: 是否为基础上下文

        Returns:
            (basic_context, subcontext, hcontext) 或 单一上下文字符串
        """
        basic_context = "Here is the patient's basic information:\n"

        if self.dataset_name == 'esrd':
            basic_data = pd.read_pickle(os.path.join(self.dataset_dir, 'basic.pkl'))[patient_id]
            gender = "male" if basic_data["Gender"] == 1 else "female"
            age = basic_data["Age"]
            if " " in basic_data["Origin_disease"]:
                ori_disease = basic_data["Origin_disease"].split(" ")[0]
                ori_disease = original_disease.get(ori_disease, ori_disease)
            else:
                ori_disease = original_disease.get(basic_data["Origin_disease"], basic_data["Origin_disease"])
            basic_disease = [disease_english[key] for key in disease_english.keys() if basic_data.get(key, 0) == 1]
            basic_disease = ", and basic disease " + ", ".join(basic_disease) if len(basic_disease) > 0 else ""
            basic_context += f"This {gender} patient, aged {age}, is an End-Stage Renal Disease(ESRD) patient with original disease {ori_disease}{basic_disease}.\n"

        elif self.dataset_name == 'cdsl':
            basic_data = pd.read_pickle(os.path.join(self.dataset_dir, 'basic.pkl'))[patient_id]
            gender = "male" if basic_data["Sex"] == 1 else "female"
            age = basic_data["Age"]
            basic_context += f"This {gender} patient, aged {age}, is an patient admitted with a diagnosis of COVID-19 or suspected COVID-19 infection.\n"

        elif self.dataset_name in ['mimic-iv', 'mimic-iii']:
            basic_data = pd.read_pickle(os.path.join(self.dataset_dir, 'basic.pkl'))[patient_id]
            gender = "male" if basic_data["Sex"] == 1 else "female"
            age = basic_data["Age"]
            basic_context += f"This {gender} patient, aged {age}, is an patient in Intensive Care Unit (ICU).\n"

        elif 'NCT' in self.dataset_name:
            # 处理NCT临床试验数据
            patient_data_path = f"/home/chunyu.lei/DEAP/SELF_DATA/mas/m-agents/processed_data/{self.dataset_name}/{patient_id}.json"
            if os.path.exists(patient_data_path):
                with open(patient_data_path, encoding='utf-8') as f:
                    patient_data = json.load(f)
                basic_context = patient_data.get('demographics', '')

                if self.dataset_name in ['NCT00003299', 'NCT03041311', 'NCT00041119', 'NCT00079274']:
                    ae_context = patient_data.get('time_series', {}).get('ae', '')
                    subcontext = basic_context
                    hcontext = basic_context + '\n' + ae_context + '\n'
                    return basic_context, subcontext, hcontext

                elif self.dataset_name == 'NCT00694382':
                    ae_context = patient_data.get('time_series', {}).get('ae', '')
                    ex_context = patient_data.get('time_series', {}).get('ex', '')
                    vs_context = patient_data.get('time_series', {}).get('vs', '')
                    subcontext = basic_context
                    hcontext = basic_context + '\n\n' + ae_context + '\n\n' + ex_context + '\n\n' + vs_context + '\n'
                    return basic_context, subcontext, hcontext

                elif self.dataset_name == 'NCT00174655':
                    ae_context = patient_data.get('time_series', {}).get('ae', '')
                    vs_context = patient_data.get('time_series', {}).get('vital', '')
                    ekg_context = patient_data.get('time_series', {}).get('ekg', '')
                    lab_context = patient_data.get('time_series', {}).get('labs', '')
                    subcontext = basic_context
                    hcontext = basic_context + '\n\n' + ae_context + '\n\n' + vs_context + '\n\n' + ekg_context + '\n\n' + lab_context + '\n'
                    return basic_context, subcontext, hcontext

        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        _, raw_x, record_time, features, lab_features, note, total_y, y, important_features = get_data_from_files(
            self.dataset_dir, self.task, self.model_name, self.mode, patient_index
        )

        if self.dataset_name in ['mimic-iv', 'mimic-iii']:
            raw_x = np.array(raw_x)[:, 2:]
            basic_context += f"Here is the clinical note of the patient:\n{note[:1024]}\n"
        elif self.dataset_name == 'esrd':
            raw_x = np.array(raw_x)
        elif self.dataset_name == 'cdsl':
            raw_x = np.array(raw_x)[:, 2:]
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        ehr_context = "Here is multivariate time-series electronic health record data of the patient, a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as sequences of numerical values for each feature.\n" + format_input_ehr(raw_x, features, lab_features, record_time)

        if is_baseline:
            return basic_context + '\n' + ehr_context

        last_visit = f"The mortality prediction risk for the patient from {self.model_name} model is {round(float(y), 2)} out of 1.0. The risk is {get_distribution(y, total_y)} among all ESRD patients. Our model especially pays great attention to the following features:\n"

        survival_stats = pd.read_pickle(os.path.join(self.dataset_dir, 'survival.pkl'))
        dead_stats = pd.read_pickle(os.path.join(self.dataset_dir, 'dead.pkl'))
        for item in important_features:
            key, value = item
            survival_mean = survival_stats[key]['50%']
            dead_mean = dead_stats[key]['50%']
            key_name = medical_name.get(key, key)
            if self.dataset_name == 'esrd':
                key_unit = ' ' + medical_unit.get(key, '') if key in medical_unit else ''
            elif self.dataset_name == 'mimic-iv':
                key_unit = ' ' + mimic_unit.get(key, '') if key in mimic_unit else ''
            else:
                key_unit = ''
            last_visit += f'{key_name}: with '
            last_visit += f'importance weight of {round(float(value["attention"]), 3)} out of 1.0. '
            last_visit += f'The feature value is {round(value["value"], 2)}{key_unit}, which is {get_mean_desc(value["value"], survival_mean)} than the median value of survival patients ({round(survival_mean, 2)}{key_unit}), {get_mean_desc(value["value"], dead_mean)} than the median value of deceased patients ({round(dead_mean, 2)}{key_unit}). '
            if self.dataset_name == 'esrd':
                if key in medical_standard:
                    last_visit += f"The reference range for healthy people is {medical_standard[key][0]}{key_unit} to {medical_standard[key][1]}{key_unit}. "
                if key in medical_standard_for_esrd:
                    last_visit += f"The reference range for ESRD patients is {medical_standard_for_esrd[key]}.\n"
                else:
                    last_visit += "\n"
            elif self.dataset_name == 'mimic-iv':
                if key in mimic_range:
                    last_visit += f"The reference range is {mimic_range[key]}.\n"
                else:
                    last_visit += "\n"
            else:
                pass
        last_visit_context = last_visit + '\n'

        subcontext = basic_context + last_visit_context
        hcontext = basic_context + '\n' + ehr_context + '\n' + last_visit_context

        return basic_context, subcontext, hcontext


__all__ = [
    'get_data_from_files',
    'get_var_desc',
    'get_trend_desc',
    'get_recommended_trend_desc',
    'get_range_desc',
    'get_mean_desc',
    'get_death_desc',
    'get_distribution',
    'format_input_ehr',
    'generate_prompt',
    'ContextBuilder',
]
