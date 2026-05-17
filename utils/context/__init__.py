"""
TrialAgents 医疗上下文模块

提供患者医疗上下文的构建和管理功能
"""

from .datasets_info import (
    medical_char,
    medical_name,
    medical_unit,
    medical_show,
    medical_standard,
    medical_standard_for_esrd,
    medical_keys,
    disease_chinese,
    disease_english,
    original_disease,
    mimic_range,
    mimic_unit,
)

from .healthcare_context_utils import (
    get_data_from_files,
    get_var_desc,
    get_trend_desc,
    get_recommended_trend_desc,
    get_range_desc,
    get_mean_desc,
    get_death_desc,
    get_distribution,
    format_input_ehr,
    generate_prompt,
    ContextBuilder,
)

from .process_data import (
    generate,
    load_synthetic,
    convert_dict_to_df,
    process_patient_df,
    process_patient_df_as_dict,
    load_clinical_trial_datasets,
    clean_bytes_string,
)

__all__ = [
    # datasets_info
    'medical_char',
    'medical_name',
    'medical_unit',
    'medical_show',
    'medical_standard',
    'medical_standard_for_esrd',
    'medical_keys',
    'disease_chinese',
    'disease_english',
    'original_disease',
    'mimic_range',
    'mimic_unit',
    # healthcare_context_utils
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
    # process_data
    'generate',
    'load_synthetic',
    'convert_dict_to_df',
    'process_patient_df',
    'process_patient_df_as_dict',
    'load_clinical_trial_datasets',
    'clean_bytes_string',
]
