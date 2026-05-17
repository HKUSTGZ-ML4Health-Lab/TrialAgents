from datetime import datetime
import json
import os
import pickle
import pandas as pd
import torch
import numpy as np

from utils.context.process_data import *
from utils.data_utils.paths import get_processed_data_path


def serialize_time_series_to_tokens(dataset, patient_list):
    # Converting dataframe to a structured string representation
    rows = []
    if dataset == 'NCT00003299' or dataset == 'NCT00079274':
        for i in range(len(patient_list[0])):
            event, value, time = patient_list[0][i], patient_list[1][i], patient_list[2][i]
            rows.append(f"Adverse Event: {event}, Severity: {value}, Day: {time}")

        return {"ae": ". ".join(rows)}

    elif dataset == 'NCT03041311':
        for i in range(len(patient_list[0])):
            event, value, start_time, end_time = patient_list[0][i], patient_list[1][i], patient_list[2][i], patient_list[3][i]
            end_time = end_time if not np.isnan(end_time) else "unknown"
            rows.append(f"Adverse Event: {event}, Severity: {value}, Start Day: {start_time}, End Day: {end_time}")

        return {"ae": ". ".join(rows)}

    elif dataset == 'NCT00041119':
        rows = []
        for i in range(len(patient_list[0])):
            event, value, time = patient_list[0][i], patient_list[1][i], patient_list[2][i]
            rows.append(f"Adverse Event: {event}, Severity: {value}, Day: {time}")

        return {"ae": ". ".join(rows)}

    elif dataset == 'NCT00694382':
        ts_dict = {
            'ae': 'Adverse Event',
            'ex': 'Drug Exposure',
            'vs': 'Vital Sign'
        }
        ae_rows, ex_rows, vs_rows = [], [], []
        for ts_index, ts_name in enumerate(['ae', 'ex', 'vs']):
            for i in range(len(patient_list[ts_index][0])):
                if ts_name == 'ae':
                    event, value, start_time, end_time = patient_list[ts_index][0][i], patient_list[ts_index][1][i], patient_list[ts_index][2][i], patient_list[ts_index][3][i]
                    end_time = end_time if not np.isnan(end_time) else "unknown"
                    serverity = 'mild' if value == 1 else 'moderate' if value == 2 else 'severe' if value == 3 else 'unknown'
                    ae_rows.append(f"{ts_dict[ts_name]}: {event}, Severity: {serverity}, Start Time: {start_time}, End Time: {end_time}")
                elif ts_name == 'ex':
                    event, value, start_time, end_time = patient_list[ts_index][0][i], patient_list[ts_index][1][i], patient_list[ts_index][2][i], patient_list[ts_index][3][i]
                    end_time = end_time if not np.isnan(end_time) else "unknown"
                    ex_rows.append(f"{ts_dict[ts_name]}: {event}, Dose: {value}, Start Time: {start_time}, End Time: {end_time}")
                elif ts_name == 'vs':
                    event, value, time = patient_list[ts_index][0][i], patient_list[ts_index][1][i], patient_list[ts_index][2][i]
                    vs_rows.append(f"{ts_dict[ts_name]}: {event}, Standardized Value: {value}, Time: {time}")
        return {
            'ae': ". ".join(ae_rows),
            'ex': ". ".join(ex_rows),
            'vs': ". ".join(vs_rows)
        }

    elif dataset == 'NCT00174655':
        ts_dict = {
            'ae': 'Adverse Event',
            'vital': 'Vital Sign',
            'ekg': 'Ekg Interpretation',
            # 'feb': 'Febrile Neutropenia',
            'labs': 'Lab Test'
        }
        ae_rows, vital_rows, ekg_rows, labs_rows = [], [], [], []
        for ts_index, ts_name in enumerate(['ae', 'vital', 'ekg', 'labs']):
            for i in range(len(patient_list[ts_index][0])):
                if ts_name == 'ae':
                    event, value, vis_day, vis_name = patient_list[ts_index][0][i], patient_list[ts_index][1][i], patient_list[ts_index][2][i], patient_list[ts_index][3][i]
                    vis_name = vis_name if vis_name else "unknown"
                    ae_rows.append(f"{ts_dict[ts_name]}: {event}, Severity: {value}, Visit Day: {vis_day}.")  # , Visit Name: {vis_name}
                elif ts_name == 'vital':
                    event, _, vis_day, vis_name = patient_list[ts_index][0][i], patient_list[ts_index][1][i], patient_list[ts_index][2][i], patient_list[ts_index][3][i]
                    vis_name = vis_name if vis_name else "unknown"
                    vital_rows.append(f"{ts_dict[ts_name]}: {event}, Visit Day: {vis_day}")
                elif ts_name == 'ekg':
                    event, _, vis_day, vis_name = patient_list[ts_index][0][i], patient_list[ts_index][1][i], patient_list[ts_index][2][i], patient_list[ts_index][3][i]
                    vis_name = vis_name if vis_name else "unknown"
                    ekg_rows.append(f"{ts_dict[ts_name]}: {event}, Visit Day: {vis_day}")
                elif ts_name == 'labs':
                    event, _, vis_day, vis_name = patient_list[ts_index][0][i], patient_list[ts_index][1][i], patient_list[ts_index][2][i], patient_list[ts_index][3][i]
                    vis_name = vis_name if vis_name else "unknown"
                    labs_rows.append(f"{ts_dict[ts_name]}: {event}, Visit Day: {vis_day}")
        return {
            'ae': ". ".join(ae_rows),
            'vital': ". ".join(vital_rows),
            'ekg': ". ".join(ekg_rows),
            'labs': ". ".join(labs_rows)
        }


def serialize_demo_info_to_tokens(dataset, demo_info):
    """
    Clinical note style for medical professionals
    """
    if dataset == 'NCT00003299':
        age = float(demo_info[0]) if not np.isnan(demo_info[0]) else "Unknown"
        gender = 'male' if demo_info[1] == 1.0 else 'female'
        ecog_ps = int(demo_info[2]) if not np.isnan(demo_info[2]) else "Unknown"
        weight_loss = 'with significant weight loss (>5-10%)' if demo_info[3] == 1.0 else 'without significant weight loss'
        num_metastases = int(demo_info[4]) if not np.isnan(demo_info[4]) else "unknown number of"
        sclc_stage = int(demo_info[5]) if not np.isnan(demo_info[5]) else "unknown stage"
        radiotherapy = 'has received' if demo_info[6] == 1.0 else 'has not received'
        chemo_cycles = int(demo_info[7]) if not np.isnan(demo_info[7]) else "unknown number of"

        return f"""This is a {age}-year-old {gender} patient with {sclc_stage} small cell lung cancer and {num_metastases} metastatic sites. The patient has an ECOG performance status of {ecog_ps} and presents {weight_loss}. Treatment history includes {chemo_cycles} completed chemotherapy cycles and the patient {radiotherapy} radiotherapy."""

    elif dataset == 'NCT03041311':
        age = demo_info[0]
        gender = 'male' if demo_info[1] == 'M' else 'female' if demo_info[1] == 'F' else 'unknown gender'
        height_cm = float(demo_info[2]) if not np.isnan(demo_info[2]) else "unknown"
        weight_kg = float(demo_info[3]) if not np.isnan(demo_info[3]) else "unknown"
        ecog_ps = int(demo_info[4]) if not np.isnan(demo_info[4]) else "unknown"
        # ecog_score = int(demo_info[5]) if not np.isnan(demo_info[5]) else "unknown"
        smk_status = demo_info[6]  # Smoking status
        smk_stage = int(demo_info[7]) if not np.isnan(demo_info[7]) else "unknown"
        smk_age = int(demo_info[8]) if not np.isnan(demo_info[8]) else "unknown"

        # Calculate BMI if height and weight are available
        if height_cm != "unknown" and weight_kg != "unknown" and height_cm > 0:
            bmi = round(weight_kg / ((height_cm/100) ** 2), 1)
            bmi_text = f" with BMI of {bmi}"
        else:
            bmi_text = ""

        smoking_text = f"The patient is a {smk_status}"
        if smk_age != "unknown":
            smoking_text += f" (started smoking at age {smk_age})"
        if smk_stage != "unknown":
            smoking_text += f" with smoking stage {smk_stage}"
        smoking_text += "."

        return f"""This is a {age}-year-old {gender} patient{bmi_text}. Physical measurements include height of {height_cm} cm and weight of {weight_kg} kg. The patient has an ECOG performance status of {ecog_ps}. {smoking_text}"""

    elif dataset == 'NCT00694382':
        age = demo_info[0] if demo_info[0] else "unknown"
        sex = 'male' if demo_info[1] == 'M' else 'female' if demo_info[1] == 'F' else 'unknown gender'
        race = demo_info[2] if demo_info[2] else "unknown race"
        region = demo_info[3] if demo_info[3] else "unknown region"
        complt = 'yes' if demo_info[-1] == 'Y' else 'no' if demo_info[-1] == 'N' else "unknown completion status"

        return  f"""This is a {age}-year-old {sex} patient. The patient identifies as {race} and is from the {region}. The completion status of the study is {complt}. """

    elif dataset == 'NCT00041119':
        Ethnicity = 'unknown ethnicity' if np.isnan(demo_info[0]) else (
            'Hispanic or Latino' if int(demo_info[0]) == 1 else 'Non-Hispanic' if int(demo_info[0]) == 2 else 'Unknown')
        Group = 'unknown group' if np.isnan(demo_info[1]) else (
            'Alliance for Clinical Trials in Oncology' if int(demo_info[1]) == 1 else 'Cancer Trials Support Unit')
        Race = 'unknown race' if np.isnan(demo_info[2]) else (
            'White' if int(demo_info[2]) == 1 else 'Black or African American' if int(demo_info[2]) == 3 else 'Asian' if int(demo_info[2]) == 4 else 'Native Hawaiian or Pacific Islander or American Indian or Alaska Native' if int(demo_info[2]) == 5 else 'Unknown')
        Menopause_Status = 'unknown menopausal status' if np.isnan(demo_info[3]) else (
            'pre-menopausal' if int(demo_info[3]) == 1 else 'post-menopausal' if int(demo_info[3]) == 2 else 'unknown menopausal status')
        Receptor_Status = 'unknown receptor status' if np.isnan(demo_info[4]) else (
            'receptor positive or unknown' if int(demo_info[4]) == 1 else 'receptor negative' if int(demo_info[4]) == 2 else 'unknown receptor status')
        Her2_neu_Status = 'unknown HER2 status' if np.isnan(demo_info[5]) else (
            'HER2 positive' if int(demo_info[5]) == 1 else 'HER2 negative' if int(demo_info[5]) == 2 else 'unknown HER2 status')
        Eligibility = 'unknown eligibility' if np.isnan(demo_info[6]) else (
            'ineligible' if int(demo_info[6]) == 1 else 'eligible' if int(demo_info[6]) == 2 else 'pending eligibility')
        Tumor_Laterality = 'unknown laterality' if np.isnan(demo_info[7]) else (
            'left breast' if int(demo_info[7]) == 1 else 'right breast' if int(demo_info[7]) == 2 else 'bilateral')
        Receptor_Status_ER = 'unknown ER status' if np.isnan(demo_info[8]) else (
            'ER negative' if int(demo_info[8]) == 1 else 'ER positive' if int(demo_info[8]) == 2 else 'unknown ER status')
        Receptor_Status_PgR = 'unknown PgR status' if np.isnan(demo_info[9]) else (
            'PgR negative' if int(demo_info[9]) == 1 else 'PgR positive' if int(demo_info[9]) == 2 else 'unknown PgR status')
        Histologic_Grade = 'unknown histologic grade' if np.isnan(demo_info[10]) else (
            'low grade' if int(demo_info[10]) == 1 else 'intermediate grade' if int(demo_info[10]) == 2 else 'high grade')
        Prior_Hormonal_Therapy = 'unknown prior hormonal therapy' if np.isnan(demo_info[11]) else (
            'no prior hormonal therapy' if int(demo_info[11]) == 1 else 'prior hormonal therapy' if int(demo_info[11]) == 2 else 'unknown')
        HT_Tamoxifen = 'unknown tamoxifen' if np.isnan(demo_info[12]) else (
            'no prior tamoxifen' if int(demo_info[12]) == 1 else 'prior tamoxifen' if int(demo_info[12]) == 2 else 'unknown')
        HT_Raloxifen = 'unknown raloxifene' if np.isnan(demo_info[13]) else (
            'no prior raloxifene' if int(demo_info[13]) == 1 else 'prior raloxifene' if int(demo_info[13]) == 2 else 'unknown')
        HT_Other = 'unknown other prior hormonal therapy' if np.isnan(demo_info[14]) else (
            'no other prior hormonal therapy' if int(demo_info[14]) == 1 else 'other prior hormonal therapy' if int(demo_info[14]) == 2 else 'unknown')
        Prior_Adjuvant_Chemo = 'unknown prior adjuvant chemotherapy' if np.isnan(demo_info[15]) else (
            'no prior adjuvant chemotherapy' if int(demo_info[15]) == 1 else 'prior adjuvant chemotherapy' if int(demo_info[15]) == 2 else 'unknown')
        Type_Biopsy = 'unknown biopsy type' if np.isnan(demo_info[16]) else (
            'core needle biopsy' if int(demo_info[16]) == 1 else 'incisional biopsy' if int(demo_info[16]) == 2 else 'excisional biopsy')
        Most_Extensive_Primary_Surgery = 'unknown surgery' if np.isnan(demo_info[17]) else (
            'partial mastectomy/lumpectomy' if int(demo_info[17]) == 1 else 'mastectomy, NOS' if int(demo_info[17]) == 2 else 'unknown surgery')
        Sentinel_Node_Biopsy = 'unknown sentinel node biopsy' if np.isnan(demo_info[18]) else (
            'no sentinel node biopsy' if int(demo_info[18]) == 1 else 'sentinel node biopsy performed' if int(demo_info[18]) == 2 else 'unknown')
        Sentinel_Node_Biopsy_Results = 'unknown sentinel node result' if np.isnan(demo_info[19]) else (
            'sentinel node negative' if int(demo_info[19]) == 1 else 'sentinel node positive' if int(demo_info[19]) == 2 else 'unknown')
        Axillary_Dissection_Performed = 'unknown axillary dissection' if np.isnan(demo_info[20]) else (
            'no axillary dissection' if int(demo_info[20]) == 1 else 'axillary dissection performed' if int(demo_info[20]) == 2 else 'unknown')
        Tumor_Size = 'unknown tumor size' if np.isnan(demo_info[21]) else (
            '<2cm' if int(demo_info[21]) == 1 else '2-5cm' if int(demo_info[21]) == 2 else '>5cm' if int(demo_info[21]) == 3 else 'unknown')
        Age_Category = 'unknown age category' if np.isnan(demo_info[22]) else (
            '20-30 years' if int(demo_info[22]) == 1 else '30-40 years' if int(demo_info[22]) == 2 else '40-50 years' if int(demo_info[22]) == 3 else '50-60 years' if int(demo_info[22]) == 4 else '60-70 years' if int(demo_info[22]) == 5 else '70+ years' if int(demo_info[22]) == 6 else 'unknown age category')
        try:
            Number_of_Positive_Axillary_Nodes = int(demo_info[23])
        except ValueError:
            Number_of_Positive_Axillary_Nodes = 'unknown'

        return (
            f"This is a patient in the {Age_Category} age group, identified as {Race} ({Ethnicity}), enrolled in the {Group}. "
            f"Key clinical information: menopausal status: {Menopause_Status}; receptor status: {Receptor_Status}; HER2 status: {Her2_neu_Status}; eligibility: {Eligibility}; "
            f"tumor laterality: {Tumor_Laterality}; tumor size: {Tumor_Size}; ER status: {Receptor_Status_ER}; PgR status: {Receptor_Status_PgR}; histologic grade: {Histologic_Grade}; "
            f"prior hormonal therapy: {Prior_Hormonal_Therapy}; tamoxifen: {HT_Tamoxifen}; raloxifene: {HT_Raloxifen}; other hormonal therapy: {HT_Other}; "
            f"prior adjuvant chemotherapy: {Prior_Adjuvant_Chemo}; biopsy type: {Type_Biopsy}; most extensive primary surgery: {Most_Extensive_Primary_Surgery}; "
            f"sentinel node biopsy: {Sentinel_Node_Biopsy}; sentinel node result: {Sentinel_Node_Biopsy_Results}; axillary dissection: {Axillary_Dissection_Performed}; "
            f"number of positive axillary nodes: {Number_of_Positive_Axillary_Nodes}."
        )

    elif dataset == 'NCT00174655':
        # ['AGE', 'SEX', 'RACE', 'CBPYN', 'HISGRADE', 'HISTYPE', 'MARGIN' ,'MULTUYN', 'PATHOM', 'PATHON', 'PATHOT', 'POSALN', 'PTSIZE', 'RESALN', 'SITE_L', 'BILOVARI', 'HORMOREP', 'HYSTEREC']
        Age = f"{demo_info[0]:.0f}-year-old" if not np.isnan(demo_info[0]) else "unknown age"
        Sex = demo_info[1] if demo_info[1] else 'unknown gender'
        Race = demo_info[2] if demo_info[2] else 'unknown race'

        return f"""This is a {Age} {Sex} patient, race: {Race} . """

    elif dataset == 'NCT00079274':
        age = demo_info[0] if demo_info[0] else "unknown"
        sex = demo_info[1] if demo_info[1] else "unknown"
        race = demo_info[2] if demo_info[2] else "unknown"
        bmi = demo_info[3] if demo_info[3] else "unknown"
        ecog = demo_info[4] if demo_info[4] else "unknown"
        histology = demo_info[5] if demo_info[5] else "unknown"
        biomarker = demo_info[6] if demo_info[6] else "unknown"

        return f"""This is a {age}-year-old {sex} patient, with a race of {race}, BMI of {bmi}, ECOG performance status of {ecog}, histology type of {histology}, and biomarker status of {biomarker}."""


def process_time_series(dataset, patient_list):
    data_dir = get_processed_data_path(dataset)
    os.makedirs(data_dir, exist_ok=True)
    patient_id = patient_list[0]
    patient_json = f"{data_dir}/{patient_id}.json"
    if os.path.exists(patient_json):
        print(f"Patient {patient_id} data already processed.")
    else:
        # Process one patient's data
        patient_id = patient_list[0]

        demo_info = patient_list[1]
        demo = serialize_demo_info_to_tokens(dataset, demo_info)

        # list_item = ['event', 'value', 'time']
        data = serialize_time_series_to_tokens(dataset, patient_list[2:-1])
        # label = patient_list[-1]

        # Convert to JSON
        # json_data = json.dumps(data, indent=2)
        # Create a combined dictionary with both adverse events and demographic information
        combined_data = {
            "time_series": data,
            "demographics": demo
        }

        # Convert the combined data to JSON and save to a file
        combined_json = json.dumps(combined_data, indent=2)
        with open(patient_json, "w") as file:
            file.write(combined_json)
        print(f"Processed data for patient {patient_id} and saved to {patient_json}")


def preprocess_data(patients_list, max_length=None, num_embeddings=None, events2ind=None, ind2events=None, mean_time=None, std_time=None):
    # ================== data processing ==================
    for i in range(len(patients_list)):
        patients_list[i][0].append('[END]')
        patients_list[i][1].append(patients_list[i][1][-1])
        patients_list[i][2].append(patients_list[i][2][-1])

    if max_length is None:
        max_length = max([len(patients_list[i][0]) for i in range(len(patients_list))])

    unique_events = np.unique([_ for i in range(len(patients_list)) for _ in patients_list[i][0]])
    if num_embeddings is None:
        num_embeddings = len(unique_events) # padding is automatically done
    if events2ind is None:
        events2ind = {unique_events[i]: i for i in range(len(unique_events))}
    if ind2events is None:
        ind2events = {v:k for k,v in events2ind.items()}
    if mean_time is None:
        mean_time = np.mean([np.mean(patients_list[i][2]) for i in range(len(patients_list))])
        std_time = np.mean([np.std(patients_list[i][2]) for i in range(len(patients_list))])

    for i in range(len(patients_list)):
        patients_list[i][0] = torch.Tensor([events2ind[_]+1 for _ in patients_list[i][0]] + [0]*(max_length-len(patients_list[i][0]))).long()
        patients_list[i][1] = torch.Tensor(patients_list[i][1] + [0]*(max_length-len(patients_list[i][1])))
        patients_list[i][2] = torch.Tensor(patients_list[i][2] + [0]*(max_length-len(patients_list[i][2])))
        patients_list[i][2] = patients_list[i][2] / std_time

    print('num_patient', len(patients_list), 'max_length', max_length)
    return patients_list, max_length, num_embeddings, events2ind, ind2events, 0, std_time


def wrap_data(config):
    """Wrap data for the collaboration pipeline.

    Uses `get_processed_data_path()` from paths.py for data directories.
    Override with environment variable: TRIALAGENTS_PROCESSED_PATH
    """
    # Create sample data
    patients_data_dir = get_processed_data_path(config['dataset_name'])
    os.makedirs(patients_data_dir, exist_ok=True)
    if config["dataset_name"] in ('NCT00003299', 'NCT03041311', 'NCT00041119', 'NCT00079274'):
        patients_data_path = os.path.join(patients_data_dir, "patients_list.pkl")
        if os.path.exists(patients_data_path):
            with open(patients_data_path, "rb") as f:
                patients_list = pickle.load(f)
            print(f"Loaded processed data from {patients_data_path}")
        else:
            data_origin = load_clinical_trial_datasets(trial_path=config["trial_path"], dataset_to_process=config["dataset_name"])
            data_raw = data_origin[0]
            data_args = {
                'indicator_col': data_raw['indicator_col'],
                'value_col': data_raw['value_col'],
                'patient_id_col': data_raw['patient_id_col'],
                'time_col': data_raw['time_col'],
                'label_col': data_raw['label_col'],
                'demographic_cols': data_raw['demographic_cols'],
            }
            # Create dataframe
            df = pd.DataFrame(data_raw['df'])
            patients_list = process_patient_df(df, patient_id_col=data_args['patient_id_col'],
                                        indicator_col=data_args['indicator_col'],
                                        value_col=data_args['value_col'],
                                        time_col=data_args['time_col'],
                                        label_col=data_args['label_col'],
                                        demographic_cols=data_args['demographic_cols'])
            # Save processed data
            with open(patients_data_path, "wb") as f:
                pickle.dump(patients_list, f)
            print(f"Processed data saved to {patients_data_path}")

    elif config["dataset_name"] == 'NCT00694382':
        patients_data_path = os.path.join(patients_data_dir, "patients_list.pkl")
        if os.path.exists(patients_data_path):
            with open(patients_data_path, "rb") as f:
                patients_list = pickle.load(f)
            print(f"Loaded processed data from {patients_data_path}")
        else:
            data_origin = load_clinical_trial_datasets(trial_path=config["trial_path"], dataset_to_process=config["dataset_name"])
            # Process ['ae', 'ex', 'vs'] dataframes
            df_names = ['ae', 'ex', 'cm', 'vs']
            patients_data = {}

            for i, df_name in enumerate(df_names):
                if df_name == 'cm':
                    continue

                data_raw = data_origin[i]
                data_args = {
                    'indicator_col': data_raw['indicator_col'],
                    'value_col': data_raw['value_col'],
                    'patient_id_col': data_raw['patient_id_col'],
                    'time_col': data_raw['time_col'],
                    'label_col': data_raw['label_col'],
                    'demographic_cols': data_raw['demographic_cols'],
                }

                df = pd.DataFrame(data_raw['df'])
                patients_single_df_list = process_patient_df(
                    df,
                    patient_id_col=data_args['patient_id_col'],
                    indicator_col=data_args['indicator_col'],
                    value_col=data_args['value_col'],
                    time_col=data_args['time_col'],
                    label_col=data_args['label_col'],
                    demographic_cols=data_args['demographic_cols']
                )

                patients_data[df_name] = {
                    'data': patients_single_df_list,
                    'patient_ids': {p[0] for p in patients_single_df_list}
                }
                print(f"Processed {df_name} data, number of patients: {len(patients_single_df_list)}")

            # Find common patient IDs across all dataframes
            common_patient_ids = patients_data['ae']['patient_ids']
            for df_name in ['ex', 'vs']:
                common_patient_ids &= patients_data[df_name]['patient_ids']

            # Create patient lookup dictionaries for efficient access
            patient_lookup = {}
            for df_name in ['ae', 'ex', 'vs']:
                patient_lookup[df_name] = {p[0]: p for p in patients_data[df_name]['data']}

            # Merge data for common patients
            patients_list = []
            for patient_id in sorted(common_patient_ids):
                ae_data = patient_lookup['ae'][patient_id]
                ex_data = patient_lookup['ex'][patient_id]
                vs_data = patient_lookup['vs'][patient_id]

                # Merge patient data: [patient_id, demographics, ae_events, ex_events, vs_events, label]
                merged_patient = [
                    patient_id,
                    ae_data[1],  # demographics (same across all dataframes)
                    [ae_data[2], ae_data[3], ae_data[4], ae_data[5]],  # ae events, values, times
                    [ex_data[2], ex_data[3], ex_data[4], ex_data[5]],  # ex events, values, times
                    [vs_data[2], vs_data[3], vs_data[4]],  # vs events, values, times
                    ae_data[-1]  # label
                ]
                patients_list.append(merged_patient)
            # Save processed data
            with open(patients_data_path, "wb") as f:
                pickle.dump(patients_list, f)
            print(f"Processed data saved to {patients_data_path}")

    elif config["dataset_name"] == 'NCT00174655':
        patients_data_path = os.path.join(patients_data_dir, "patients_list.pkl")
        if os.path.exists(patients_data_path):
            with open(patients_data_path, "rb") as f:
                patients_list = pickle.load(f)
            print(f"Loaded processed data from {patients_data_path}")
        else:
            data_origin = load_clinical_trial_datasets(trial_path=config["trial_path"], dataset_to_process=config["dataset_name"])
            df_names = ['ae', 'vital', 'ekg', 'labs']  # 'feb'
            patients_data = {}

            for i, df_name in enumerate(df_names):
                if df_name == 'cm':
                    continue

                data_raw = data_origin[i]
                data_args = {
                    'indicator_col': data_raw['indicator_col'],
                    'value_col': data_raw['value_col'],
                    'patient_id_col': data_raw['patient_id_col'],
                    'time_col': data_raw['time_col'],
                    'label_col': data_raw['label_col'],
                    'demographic_cols': data_raw['demographic_cols'],
                }

                df = pd.DataFrame(data_raw['df'])
                patients_single_df_list = process_patient_df(
                    df,
                    patient_id_col=data_args['patient_id_col'],
                    indicator_col=data_args['indicator_col'],
                    value_col=data_args['value_col'],
                    time_col=data_args['time_col'],
                    label_col=data_args['label_col'],
                    demographic_cols=data_args['demographic_cols']
                )

                patients_data[df_name] = {
                    'data': patients_single_df_list,
                    'patient_ids': {p[0] for p in patients_single_df_list}
                }
                print(f"Processed {df_name} data, number of patients: {len(patients_single_df_list)}")

            # Find common patient IDs across all dataframes
            common_patient_ids = patients_data['ae']['patient_ids']
            for df_name in ['vital', 'ekg', 'labs']:
                common_patient_ids &= patients_data[df_name]['patient_ids']

            # Create patient lookup dictionaries for efficient access
            patient_lookup = {}
            for df_name in ['ae', 'vital', 'ekg', 'labs']:
                patient_lookup[df_name] = {p[0]: p for p in patients_data[df_name]['data']}

            # Merge data for common patients
            patients_list = []
            for patient_id in sorted(common_patient_ids):
                ae_data = patient_lookup['ae'][patient_id]
                vs_data = patient_lookup['vital'][patient_id]
                ekg_data = patient_lookup['ekg'][patient_id]
                labs_data = patient_lookup['labs'][patient_id]

                # Merge patient data: [patient_id, demographics, ae_events, vital_events, ekg_events, labs_events, label]
                merged_patient = [
                    patient_id,
                    vs_data[1],  # demographics (same across all dataframes)
                    [ae_data[2], ae_data[3], ae_data[4], ae_data[5]],  # ae events, values, times
                    [vs_data[2], vs_data[3], vs_data[4], vs_data[5]],  # vs events, values, times
                    [ekg_data[2], ekg_data[3], ekg_data[4], ekg_data[5]],  # ekg events, values, times
                    [labs_data[2], labs_data[3], labs_data[4], labs_data[5]],  # labs events, values, times
                    vs_data[-1]  # label
                ]
                patients_list.append(merged_patient)
            # Save processed data
            with open(patients_data_path, "wb") as f:
                pickle.dump(patients_list, f)
            print(f"Processed data saved to {patients_data_path}")
    print(f"Total patients after processing: {len(patients_list)}")
    return patients_list
