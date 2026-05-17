nct_config = {
    "dataset_name": "NCT00003299",
    "llm": "deepseek-chat",
    "group": "experimental",
    "task": "MRP",
    "trial_path": "/home/chunyu.lei/DEAP/SELF_DATA/mas/kbvae/data",
    "main_metric": ["auprc", "auroc", "bacc", "wf1"],
    "loop_num": 1,
    "tau": 0.75,
    "fold": 5,
    "start_idx": 0,
    "end_idx": 100,
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",
    "clinician_num": 3,
    "max_round": 3,
    "model_names": ['Transformer', 'LSTM', 'GRU'],
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "epochs": 20,
    "patience": 5,
    "d_model": 256,
    "n_head": 8,
    "n_layer": 1,
    "dropout": 0.2,
    "T": 5.0,
    "mode": "test",
    "gpuid": "6",
}

dataset_subjects = {
    "NCT00003299": 561,
    "NCT03041311": 51,
    "NCT00694382": 1159,
    "NCT00041119": 932,
    "NCT00174655": 974,
    "NCT00079274": 524,
}

config = [
    nct_config,
]

__all__ = [
    'nct_config',
    'dataset_subjects',
    'config',
]
