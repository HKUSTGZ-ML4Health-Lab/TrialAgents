"""
DivideMix Training Core for TrialAgents

Migrated from m-agents/utils/lnl_train.py
Refactored to remove module-level side effects and hardcoded paths.
"""
from __future__ import print_function
import os
import argparse
import sys
import random
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from baselines.small_models import *
from utils.retrieval.hparams import nct_config as default_config
from utils.data_utils.paths import get_checkpoints_path, get_logs_path
from src.training.lnl_dataloader import nct_dataloader


# Module-level state: initialized lazily in run_dividemix()
_args = None
_stats_log = None
_test_log = None
_train_num = None
_test_loader = None
_eval_loader = None
_et_loader = None
_writer = None  # TensorBoard writer
_CE = None
_CEloss = None
_conf_penalty = None
_criterion = None
_warm_up = None
_device = None


def create_parser():
    parser = argparse.ArgumentParser(description='DivideMix Training')
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--noise_mode', default='no')
    parser.add_argument('--warm_up', default=5, type=int, help='warm up epochs')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=5, type=float, choices=[0.5, 5.0, 10.0], help='sharpening temperature')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--r', default=0.0, type=float, help='noise ratio')
    parser.add_argument('--id', default='')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--gpuid', default=7, type=int)
    parser.add_argument('--num_class', default=2, type=int)
    return parser


def _build_args(config=None, cli_args=None):
    """Build args from config dict and optional CLI args, without side effects."""
    if config is None:
        config = default_config

    if cli_args is None:
        parser = create_parser()
        cli_args = parser.parse_args([])  # parse empty list for defaults

    cli_args.lr = config.get("learning_rate", cli_args.lr)
    cli_args.p_threshold = config.get("tau", cli_args.p_threshold)
    cli_args.num_epochs = config.get("epochs", cli_args.num_epochs)
    cli_args.patience = config.get("patience", 5)
    cli_args.data_path = config.get("trial_path", "")
    cli_args.dataset = config.get("dataset_name", "")
    cli_args.batch_size = config.get("batch_size", cli_args.batch_size)
    cli_args.gpuid = int(config.get("gpuid", cli_args.gpuid))
    cli_args.T = float(config.get("T", cli_args.T))
    cli_args.group = config.get("group", "experimental")
    cli_args.model_name = config.get("model_names", ["Transformer"])[0]

    # Handle GPU
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        cli_args.gpuid = -1
    else:
        num_gpus = torch.cuda.device_count()
        if cli_args.gpuid >= num_gpus or cli_args.gpuid < 0:
            cli_args.gpuid = 0 if num_gpus > 0 else -1
        if cli_args.gpuid >= 0:
            torch.cuda.set_device(cli_args.gpuid)

    np.random.seed(cli_args.seed)
    random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    if torch.cuda.is_available() and cli_args.gpuid >= 0:
        torch.cuda.manual_seed_all(cli_args.seed)

    return cli_args


def _get_device(args):
    d = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() and args.gpuid >= 0 else 'cpu')
    return d


def _to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor


# ---- Training Functions ----

def train(fold, epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader,
          args, criterion, warm_up, writer, device):
    net.train()
    net2.eval()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, labeled_batch in enumerate(labeled_trainloader):
        try:
            unlabeled_batch = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            unlabeled_batch = next(unlabeled_train_iter)

        inputs_x = labeled_batch['features1']
        inputs_x2 = labeled_batch['features2']
        labels_x = labeled_batch['target']
        w_x = labeled_batch['prob']
        seq_len_x1 = labeled_batch['seq_len1']
        seq_len_x2 = labeled_batch['seq_len2']

        inputs_u = unlabeled_batch['features1']
        inputs_u2 = unlabeled_batch['features2']
        seq_len_u1 = unlabeled_batch['seq_len1']
        seq_len_u2 = unlabeled_batch['seq_len2']

        batch_size = inputs_x.size(0)

        inputs_x = _to_device(inputs_x, device).float()
        inputs_x2 = _to_device(inputs_x2, device).float()
        labels_x = _to_device(labels_x, device).float()
        w_x = _to_device(w_x, device).float()
        inputs_u = _to_device(inputs_u, device).float()
        inputs_u2 = _to_device(inputs_u2, device).float()
        seq_len_x1 = _to_device(seq_len_x1, device)
        seq_len_x2 = _to_device(seq_len_x2, device)
        seq_len_u1 = _to_device(seq_len_u1, device)
        seq_len_u2 = _to_device(seq_len_u2, device)

        with torch.no_grad():
            batch_u1 = {'features': inputs_u, 'seq_len': seq_len_u1}
            batch_u2 = {'features': inputs_u2, 'seq_len': seq_len_u2}
            outputs_u11 = net(batch_u1)
            outputs_u12 = net(batch_u2)
            outputs_u21 = net2(batch_u1)
            outputs_u22 = net2(batch_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)
            targets_u = ptu / ptu.sum(dim=-1, keepdim=True)
            targets_u = targets_u.detach()

            batch_x1 = {'features': inputs_x, 'seq_len': seq_len_x1}
            batch_x2 = {'features': inputs_x2, 'seq_len': seq_len_x2}
            outputs_x = net(batch_x1)
            outputs_x2 = net(batch_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            if labels_x.dim() == 1:
                labels_x_onehot = torch.zeros(batch_size, 2).to(device)
                labels_x_binary = (labels_x > 0.5).long()
                labels_x_onehot.scatter_(1, labels_x_binary.view(-1, 1), 1)
            elif labels_x.dim() == 2 and labels_x.size(1) == 2:
                labels_x_onehot = labels_x
            else:
                labels_x_onehot = torch.zeros(batch_size, 2).to(device)
                labels_x_binary = labels_x.long()
                labels_x_onehot.scatter_(1, labels_x_binary.view(-1, 1), 1)

            px = w_x.unsqueeze(1) * labels_x_onehot + (1 - w_x).unsqueeze(1) * px
            ptx = px ** (1 / args.T)
            targets_x = ptx / ptx.sum(dim=-1, keepdim=True)
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_seq_lens = torch.cat([seq_len_x1, seq_len_x2, seq_len_u1, seq_len_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        seq_len_a, seq_len_b = all_seq_lens, all_seq_lens[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_seq_len = torch.where(torch.rand(seq_len_a.size()).to(device) < l, seq_len_a, seq_len_b)
        mixed_target = l * target_a + (1 - l) * target_b

        mixed_batch = {'features': mixed_input, 'seq_len': mixed_seq_len}
        logits = net(mixed_batch)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2],
                                 logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, warm_up)

        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.4f  Unlabeled loss: %.4f Total_Loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            Lx.item(), Lu.item(), loss.item()))
        sys.stdout.flush()

        if writer is not None:
            global_step = epoch * num_iter + batch_idx
            writer.add_scalar(f'Fold_{fold}/Train/Labeled_Loss', Lx.item(), global_step)
            writer.add_scalar(f'Fold_{fold}/Train/Unlabeled_Loss', Lu.item(), global_step)
            writer.add_scalar(f'Fold_{fold}/Train/Total_Loss', loss.item(), global_step)

    return loss


def warmup(fold, epoch, net, optimizer, dataloader, args, writer, device):
    global _conf_penalty
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, batch in enumerate(dataloader):
        inputs, seq_len, labels = batch['features'], batch['seq_len'], batch['target']
        inputs = _to_device(inputs, device).float()
        seq_len = _to_device(seq_len, device).float()
        labels = _to_device(labels, device).float()
        inputs_dict = {'features': inputs, 'seq_len': seq_len}
        optimizer.zero_grad()
        outputs = net(inputs_dict)

        if labels.dim() == 1:
            if labels.dtype.is_floating_point and ((labels >= 0) & (labels <= 1)).all():
                target_labels = (labels > 0.5).long()
            else:
                target_labels = labels.long()
            loss = F.cross_entropy(outputs, target_labels)
        elif labels.dim() == 2 and labels.size(1) == 2:
            log_probs = F.log_softmax(outputs, dim=1)
            loss = -torch.mean(torch.sum(labels * log_probs, dim=1))
        else:
            target_labels = labels.long()
            loss = F.cross_entropy(outputs, target_labels)

        if args.noise_mode == 'asym':
            penalty = _conf_penalty(outputs)
            L = loss + penalty
        else:
            L = loss
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs,
                            batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()

        if writer is not None:
            global_step = epoch * num_iter + batch_idx
            writer.add_scalar(f'Fold_{fold}/Warmup/Loss', loss.item(), global_step)


def _eval(fold, epoch, net1, net2, eval_loader, writer, device):
    net1.eval()
    net2.eval()
    total = 0
    loss = 0
    target_binary, probs_pos, predicted_binary = [], [], []

    with torch.no_grad():
        for _, batch in enumerate(eval_loader):
            inputs = batch['features']
            seq_len = batch['seq_len']
            targets = batch['target']

            inputs = _to_device(inputs, device).float()
            seq_len = _to_device(seq_len, device)
            targets = _to_device(targets, device).float()

            batch_data = {'features': inputs, 'seq_len': seq_len}
            outputs1 = net1(batch_data)
            outputs2 = net2(batch_data)
            outputs = (outputs1 + outputs2) / 2

            probs = torch.softmax(outputs, dim=-1)
            probs_pos_batch = probs[:, 1]
            predicted_binary_batch = (probs_pos_batch > 0.5).float()

            probs_pos.extend(probs_pos_batch.cpu().numpy())
            predicted_binary.extend(predicted_binary_batch.cpu().numpy())

            if targets.dim() == 1:
                target_binary.extend((targets > 0.5).float().cpu().numpy())
            elif targets.dim() == 2 and targets.size(1) == 2:
                target_binary.extend(targets.argmax(dim=1).float().cpu().numpy())
            else:
                target_binary.extend(targets.float().cpu().numpy())
            loss += F.cross_entropy(outputs, targets.long())
            total += targets.size(0)

    avg_loss = loss / total
    target_binary = np.array(target_binary)
    probs_pos = np.array(probs_pos)
    predicted_binary = np.array(predicted_binary)
    acc = accuracy_score(target_binary, predicted_binary)
    f1 = f1_score(target_binary, predicted_binary)
    if len(set(target_binary)) == 1:
        auroc, auprc = 0, 0
    else:
        auroc = roc_auc_score(target_binary, probs_pos)
        auprc = average_precision_score(target_binary, probs_pos)

    print("\n| Eval Epoch #%d\t ACC: %.4f, AUROC: %.4f" % (epoch, acc, auroc))
    if writer is not None:
        global_step = epoch
        writer.add_scalar(f'Fold_{fold}/Eval/ACC', acc, global_step)
        writer.add_scalar(f'Fold_{fold}/Eval/F1', f1, global_step)
        writer.add_scalar(f'Fold_{fold}/Eval/AUROC', auroc, global_step)
        writer.add_scalar(f'Fold_{fold}/Eval/AUPRC', auprc, global_step)

    return avg_loss


def _test(fold, epoch, net1, net2, test_loader, log_file, writer, device):
    def min_pse(labels, preds):
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, _ = precision_recall_curve(labels, preds)
        minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
        return minpse_score

    net1.eval()
    net2.eval()
    total = 0
    correct = 0
    target_binary, probs_pos, predicted_binary = [], [], []

    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            inputs = batch['features']
            seq_len = batch['seq_len']
            targets = batch['target']

            inputs = _to_device(inputs, device).float()
            seq_len = _to_device(seq_len, device)
            targets = _to_device(targets, device).float()

            batch_data = {'features': inputs, 'seq_len': seq_len}
            outputs1 = net1(batch_data)
            outputs2 = net2(batch_data)
            outputs = (outputs1 + outputs2) / 2

            probs = torch.softmax(outputs, dim=-1)
            probs_pos_batch = probs[:, 1]
            predicted_binary_batch = (probs_pos_batch > 0.5).float()

            probs_pos.extend(probs_pos_batch.cpu().numpy())
            predicted_binary.extend(predicted_binary_batch.cpu().numpy())

            if targets.dim() == 1:
                target_binary_batch = (targets > 0.5).float()
            elif targets.dim() == 2 and targets.size(1) == 2:
                target_binary_batch = targets.argmax(dim=1).float()
            else:
                target_binary_batch = targets.float()
            target_binary.extend(target_binary_batch.cpu().numpy())

            total += targets.size(0)
            correct += (predicted_binary_batch == target_binary_batch).sum().item()

    target_binary = np.array(target_binary)
    probs_pos = np.array(probs_pos)
    predicted_binary = np.array(predicted_binary)
    acc = accuracy_score(target_binary, predicted_binary)
    f1 = f1_score(target_binary, predicted_binary)
    if len(set(target_binary)) == 1:
        auroc, auprc, _minpse = 0, 0, 0
    else:
        auroc = roc_auc_score(target_binary, probs_pos)
        auprc = average_precision_score(target_binary, probs_pos)
        _minpse = min_pse(target_binary, probs_pos)

    print("\n| Test Epoch #%d\t ACC: %.4f, F1: %.4f,  AUROC: %.4f, AUPRC: %.4f, MinPSE: %.4f" %
          (epoch, acc, f1, auroc, auprc, _minpse))
    log_file.write('\nEpoch:%d   ACC:%.4f, F1: %.4f, AUROC:%.4f, AUPRC:%.4f, MinPSE: %.4f' %
                   (epoch, acc, f1, auroc, auprc, _minpse))
    log_file.flush()

    if writer is not None:
        writer.add_scalar(f'Fold_{fold}/Test/ACC', acc, epoch)
        writer.add_scalar(f'Fold_{fold}/Test/F1', f1, epoch)
        writer.add_scalar(f'Fold_{fold}/Test/AUROC', auroc, epoch)
        writer.add_scalar(f'Fold_{fold}/Test/AUPRC', auprc, epoch)

    preds_df = pd.DataFrame({'label': target_binary, 'pred': probs_pos})

    return {
        'acc': acc,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'minpse': _minpse,
        'preds_df': preds_df
    }


def eval_train(model, all_loss, train_num, et_loader, CE, writer, device):
    model.eval()
    losses = torch.zeros(train_num).to(device).float()
    with torch.no_grad():
        for batch_idx, batch in enumerate(et_loader):
            inputs = batch['features']
            seq_len = batch['seq_len']
            targets = batch['target']
            index = batch['index']

            inputs = _to_device(inputs, device).float()
            seq_len = _to_device(seq_len, device)
            targets = _to_device(targets, device).float()

            batch_data = {'features': inputs, 'seq_len': seq_len}
            outputs = model(batch_data)
            outputs = torch.softmax(outputs, dim=1)

            if targets.dim() == 1:
                if targets.dtype.is_floating_point and ((targets >= 0) & (targets <= 1)).all():
                    target_labels = (targets > 0.5).long()
                else:
                    target_labels = targets.long()
            elif targets.dim() == 2 and targets.size(1) == 2:
                target_labels = targets.argmax(dim=1).long()
            else:
                target_labels = targets.long()

            loss = CE(outputs, target_labels)

            for b in range(inputs.size(0)):
                idx = index[b].item() if torch.is_tensor(index[b]) else index[b]
                losses[idx] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
    all_loss.append(losses)

    input_loss = losses.reshape(-1, 1)
    input_loss = input_loss.cpu().numpy()

    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]

    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16, lambda_u=25):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class ClassificationSemiLoss(object):
    def __init__(self, class_weights=None, device=None):
        self.class_weights = class_weights
        if class_weights is not None and device is not None:
            self.class_weights = class_weights.to(device).float()
        self.device = device

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, lambda_u=25):
        if targets_x.dim() == 1:
            Lx = F.cross_entropy(outputs_x, targets_x.long())
        else:
            log_probs_x = F.log_softmax(outputs_x, dim=1)
            if self.class_weights is not None:
                weights = self.class_weights.unsqueeze(0)
                weighted_targets_x = targets_x * weights
                Lx = -torch.mean(torch.sum(weighted_targets_x * log_probs_x, dim=1))
            else:
                Lx = -torch.mean(torch.sum(targets_x * log_probs_x, dim=1))

        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up, lambda_u=lambda_u)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model(train_loader, config, device):
    all_indicators, all_seq_lens = [], []
    for batch in train_loader:
        indicators = batch['features'][:, :, 1].numpy().flatten()
        seq_lens = batch['seq_len'].numpy()
        all_indicators.extend(indicators)
        all_seq_lens.extend(seq_lens)

    all_indicators = np.array(all_indicators)
    max_indicator = int(np.ceil(all_indicators.max()))
    max_seq_len_in_data = int(np.max(all_seq_lens))
    dynamic_seq_len = max_seq_len_in_data + 3
    dynamic_vocab_size = max_indicator + 3
    print(f"Dynamic seq_len: {dynamic_seq_len}, vocab_size: {dynamic_vocab_size}")

    dataset_name = config.get('dataset_name', '')
    if dataset_name == 'NCT03041311':
        d_model = config.get('d_model', 256) // 4
        n_head = config.get('n_head', 8) // 2
    else:
        d_model = config.get('d_model', 256)
        n_head = config.get('n_head', 8)
    n_layer = config.get('n_layer', 1)
    dropout = config.get('dropout', 0.2)
    model_name = config.get('model_names', ['Transformer'])[0]

    if "Trans" in model_name:
        model = NCTTimeSeriesTransformer(
            d_model=d_model, n_heads=n_head, n_layers=n_layer,
            dropout=dropout, max_seq_len=dynamic_seq_len,
            indicator_vocab_size=dynamic_vocab_size, d_out=2,
        )
    elif "LSTM" in model_name:
        model = LSTMPredictor(
            hidden_size=d_model // 8,
            num_embeddings=dynamic_vocab_size,
            num_layers=n_layer,
        )
    elif "GRU" in model_name:
        model = NCTTimeSeriesGRU(
            indicator_vocab_size=dynamic_vocab_size,
            max_seq_len=dynamic_seq_len, hidden_size=d_model // 8,
            num_layers=n_layer, dropout=dropout, d_out=2,
            bidirectional=True,
        )
    else:
        model = NCTTimeSeriesTransformer(
            d_model=d_model, n_heads=n_head, n_layers=n_layer,
            dropout=dropout, max_seq_len=dynamic_seq_len,
            indicator_vocab_size=dynamic_vocab_size, d_out=2,
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} created, total params: {total_params:,}")
    model = model.to(device)
    return model


def run_dividemix(log_dir, fold, train_zip=None, es_zip=None, test_zip=None,
                  hyper_args=None, config=None):
    """
    Run DivideMix training for one cross-validation fold.

    Args:
        log_dir: Directory for logs and checkpoints
        fold: Fold index
        train_zip: Training data tuple (X, y)
        es_zip: Early stopping data tuple (X, y)
        test_zip: Test data tuple (X, y)
        hyper_args: Optional argparse.Namespace with hyperparameter overrides
        config: Optional config dict (defaults to nct_config from hparams)
    """
    if config is None:
        config = default_config

    # Build args
    args = _build_args(config)
    if hyper_args:
        for key, value in hyper_args.items():
            setattr(args, key, value)

    device = _get_device(args)
    print(f"Using device: {device}")

    dataset_name = args.dataset
    noise_mode = args.noise_mode
    r = args.r
    hps_suffix = f'_hps{args.hyper_trial_id}' if hasattr(args, 'hyper_trial_id') else ''

    # Checkpoint and log directories
    checkpoint_dir = get_checkpoints_path(dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    stats_log = open(f'{checkpoint_dir}/{dataset_name}_{r:.1f}_{noise_mode}_stats{hps_suffix}.txt', 'a')
    test_log = open(f'{checkpoint_dir}/{dataset_name}_{r:.1f}_{noise_mode}_acc{hps_suffix}.txt', 'a')

    os.makedirs(log_dir, exist_ok=True)
    tb_log_dir = os.path.join(log_dir, 'tb')
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    print(f'Running DivideMix with {noise_mode} noise {r:.1f}')
    print(f'Dataset: {dataset_name}, Batch: {args.batch_size}, LR: {args.lr:.4f}, '
          f'Epochs: {args.num_epochs}, Seed: {args.seed}')

    warm_up_epochs = args.warm_up
    patience = args.patience
    train_num = len(train_zip[0])

    data_path = args.data_path

    loader = nct_dataloader(
        args, num_workers=0,
        root_dir=os.path.join(data_path, dataset_name),
        train_zip=train_zip, es_zip=es_zip, test_zip=test_zip,
        log=stats_log, noise_file='', use_soft_labels=True
    )

    print('| Building nets')
    warmup_trainloader = loader.run('warmup')
    net1 = create_model(warmup_trainloader, config, device)
    net2 = create_model(warmup_trainloader, config, device)
    if device.type == 'cuda':
        cudnn.benchmark = True

    learning_rate = args.lr
    weight_decay = getattr(args, 'weight_decay', config.get('weight_decay', 1e-5))

    optimizer1 = optim.AdamW(net1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer2 = optim.AdamW(net2.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = args.num_epochs
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs)

    # Class weights
    pos_count = sum(1 for l in warmup_trainloader.dataset.train_label if l >= 0.5)
    neg_count = len(warmup_trainloader.dataset.train_label) - pos_count
    if pos_count == 0:
        pos_count = max(neg_count // 10, 1)
        neg_count -= pos_count
    elif neg_count == 0:
        neg_count = max(pos_count // 10, 1)
        pos_count -= neg_count
    total_samples = pos_count + neg_count
    neg_weight = total_samples / (2.0 * neg_count)
    pos_weight = total_samples / (2.0 * pos_count)
    class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float).to(device)
    print(f"Class distribution - Neg: {neg_count}, Pos: {pos_count}")
    print(f"Class weights - [neg: {neg_weight:.4f}, pos: {pos_weight:.4f}]")

    CE = nn.CrossEntropyLoss(reduction='none')
    criterion = ClassificationSemiLoss(class_weights=class_weights, device=device)
    conf_penalty = NegEntropy() if args.noise_mode == 'asym' else None

    all_loss = [[], []]
    best_loss = [float('inf'), float('inf')]

    for epoch in range(num_epochs + 1):
        lr = learning_rate
        lr_decay_epoch = getattr(args, 'lr_decay_epoch', 150)
        if epoch >= lr_decay_epoch and lr_decay_epoch > 0:
            lr /= 10
        for pg in optimizer1.param_groups:
            pg['lr'] = lr
        for pg in optimizer2.param_groups:
            pg['lr'] = lr

        test_loader_ = loader.run('test')
        et_loader = loader.run('eval_train')
        eval_loader_ = loader.run('eval')

        if epoch < warm_up_epochs:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(fold, epoch, net1, optimizer1, warmup_trainloader, args, writer, device)
            print('\nWarmup Net2')
            warmup(fold, epoch, net2, optimizer2, warmup_trainloader, args, writer, device)
        else:
            prob1, all_loss[0] = eval_train(net1, all_loss[0], train_num, et_loader, CE, writer, device)
            prob2, all_loss[1] = eval_train(net2, all_loss[1], train_num, et_loader, CE, writer, device)

            p_threshold = args.p_threshold
            pred1 = (prob1 > p_threshold)
            pred2 = (prob2 > p_threshold)

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)
            train(fold, epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader,
                  args, criterion, warm_up_epochs, writer, device)

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)
            train(fold, epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader,
                  args, criterion, warm_up_epochs, writer, device)

            loss1 = _eval(fold, epoch, net1, net2, eval_loader_, writer, device)
            loss2 = _eval(fold, epoch, net2, net1, eval_loader_, writer, device)
            current_loss1 = float(loss1)
            current_loss2 = float(loss2)

            improved = False
            model_name = config.get('model_names', ['default'])[0] if 'model_names' in config else 'dividemix'

            if current_loss1 < best_loss[0]:
                best_loss[0] = current_loss1
                improved = True
                torch.save(net1.state_dict(),
                           f'{checkpoint_dir}/{dataset_name}_{model_name}_fold{fold}_net1.pth')
                print(f"  ✓ Net1 improved: {current_loss1:.4f} (saved)")

            if current_loss2 < best_loss[1]:
                best_loss[1] = current_loss2
                improved = True
                torch.save(net2.state_dict(),
                           f'{checkpoint_dir}/{dataset_name}_{model_name}_fold{fold}_net2.pth')
                print(f"  ✓ Net2 improved: {current_loss2:.4f} (saved)")

            if improved:
                patience = args.patience
            else:
                patience -= 1
                print(f"  ⚠ No improvement, patience: {patience}")

            if patience <= 0:
                print(f"\n🛑 Early stopping at epoch {epoch + 1}")
                break

        _test(fold, epoch, net1, net2, test_loader_, test_log, writer, device)
        scheduler1.step()
        scheduler2.step()

    # Load best and final test
    model_name = config.get('model_names', ['default'])[0] if 'model_names' in config else 'dividemix'
    try:
        net1.load_state_dict(torch.load(f'{checkpoint_dir}/{dataset_name}_{model_name}_fold{fold}_net1.pth'))
        net2.load_state_dict(torch.load(f'{checkpoint_dir}/{dataset_name}_{model_name}_fold{fold}_net2.pth'))
    except Exception as e:
        print(f"Warning: Failed to load best model, using final. Error: {e}")
    test_results = _test(fold, epoch, net1, net2, test_loader_, test_log, writer, device)

    stats_log.close()
    test_log.close()
    if writer is not None:
        print(f'\nTensorBoard logs saved to: {log_dir}')
        writer.close()
    return test_results


if __name__ == '__main__':
    parser = create_parser()
    cli_args = parser.parse_args()
    args = _build_args(default_config, cli_args)
    run_dividemix(log_dir=get_logs_path('training'), fold=0, config=default_config)
    print('DivideMix training completed.')
