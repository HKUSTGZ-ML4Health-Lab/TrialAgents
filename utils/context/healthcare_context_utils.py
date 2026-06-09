"""
TrialAgents Healthcare Context Building Module

Migrated from m-agents/utils/healthcare_context_utils.py
Provides patient medical context generation functionality.

Dependencies:
    utils.context.datasets_info: Medical data dictionaries
    utils.context.process_data: Data processing functions
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
    Load patient data from files.

    Args:
        data_url: Data directory
        task: Task name
        model: Model name
        mode: Mode (test/val)
        patient_index: Patient index

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
    """Get variable description (as percentage)."""
    if var > 0:
        return round(var * 100, 2)
    else:
        return round(-var * 100, 2)


def get_trend_desc(var: float) -> str:
    """Get trend description."""
    if var > 0:
        return "increased"
    else:
        return "decreased"


def get_recommended_trend_desc(var: float) -> str:
    """Get recommended trend description."""
    if var > 0:
        return "decrease"
    else:
        return "increase"


def get_range_desc(key: str, var: float) -> str:
    """Get range description."""
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
    """Get description relative to the mean."""
    if var < mean:
        return f"{round((mean - var) / mean * 100, 0)}% lower"
    elif var > mean:
        return f"{round((var - mean) / mean * 100, 0)}% higher"


def get_death_desc(risk: float) -> str:
    """Get death risk level description."""
    if risk < 0.5:
        return "a low level"
    elif risk < 0.7:
        return "a high level"
    else:
        return "an extremely high level"


def get_distribution(data, values) -> str:
    """Get data distribution description."""
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
    Format EHR data into text description.

    Args:
        raw_x: Raw feature data
        features: Feature name list
        lab_features: Lab test feature list
        record_time: Record time list

    Returns:
        Formatted EHR description text
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
    Generate a medical context prompt.

    Args:
        dataset: Dataset name
        data_url: Data directory
        patient_index: Patient index
        patient_id: Patient ID

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
    Medical context builder.

    Generates medical context for LLM analysis based on patient data.
    """

    def __init__(
        self,
        dataset_name: str,
        task: str,
        model_name: str,
        mode: str
    ) -> None:
        """
        Initialize the context builder.

        Args:
            dataset_name: Dataset name (esrd, cdsl, mimic-iv, mimic-iii, NCT*)
            task: Task name
            model_name: Model name
            mode: Mode (test/val)
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
        Generate patient medical context.

        Args:
            patient_index: Patient index in the dataset
            patient_id: Patient ID
            is_baseline: Whether to generate baseline context only

        Returns:
            (basic_context, subcontext, hcontext) or a single context string
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
            # Handle NCT clinical trial data
            from pathlib import Path as _Path
            _default_dir = str(_Path(__file__).parent.parent.parent.parent / "m-agents" / "processed_data")
            _processed_dir = os.getenv("MAS_PROCESSED_DATA_DIR", _default_dir)
            patient_data_path = os.path.join(_processed_dir, self.dataset_name, f"{patient_id}.json")
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
