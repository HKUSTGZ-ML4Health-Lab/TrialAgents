"""
TrialAgents 数据运行工具

从 m-agents/utils/runner_utils.py 迁移至此
提供时间序列数据处理功能
"""

import os
import json
from typing import List


def process_time_series(dataset: str, patient_list: List) -> None:
    """
    将患者数据序列化并保存为JSON

    Args:
        dataset: 数据集名称
        patient_list: 患者数据列表

    Note:
        数据保存到 /home/chunyu.lei/DEAP/SELF_DATA/mas/m-agents/processed_data/{dataset}/{patient_id}.json
    """
    save_base = f"/home/chunyu.lei/DEAP/SELF_DATA/mas/m-agents/processed_data/{dataset}"

    if not os.path.exists(save_base):
        os.makedirs(save_base)

    for patient_data in patient_list:
        patient_id = patient_data[0]
        save_path = os.path.join(save_base, f"{patient_id}.json")

        if os.path.exists(save_path):
            continue

        # patient_data格式: [patient_id, features..., label]
        # 序列化处理
        try:
            with open(save_path, 'w') as f:
                json.dump(patient_data, f)
        except Exception as e:
            print(f"Error saving patient {patient_id}: {e}")


__all__ = ['process_time_series']
