"""
TrialAgents Medical Context Module

Patient medical context construction and management.
"""


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
