# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""

import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
import argparse
import logging
from transformers import get_linear_schedule_with_warmup

from dataset import CustomDataSet, collate_fn
from model import AttentionDTA
from prefetch_generator import BackgroundGenerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_loss_function():
    return nn.CrossEntropyLoss()


def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[:val_start]

    return trainset, validset


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def train(model, train_data_loader, eval_data_loader, test_data_loader, args):
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    criterion = get_loss_function()
    t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.epochs

    # 按作者风格：不在 AdamW 中使用 weight_decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        eps=args.adam_epsilon
    )

    if 0 < args.warmup_steps < 1:
        args.warmup_steps = int(args.warmup_steps * t_total)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    train_iterator = trange(0, int(args.epochs), desc="Epoch", disable=False)

    patience = 0
    best_score = 0.0
    global_step = 0

    for epoch in train_iterator:
        local_steps = 0
        tr_loss = 0.0
        model.train()

        epoch_iterator = tqdm(
            BackgroundGenerator(train_data_loader),
            total=len(train_data_loader),
            desc="training"
        )

        for step, data in enumerate(epoch_iterator):
            compounds, atom, node, edge, attr, proteins, compound_mask, protein_mask, labels = [
                d.cuda() for d in data
            ]

            predicts = model.forward(
                compounds, proteins, atom, node, edge, attr, compound_mask, protein_mask
            )
            loss = criterion(predicts, labels.long())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss / max(local_steps, 1))

        avg_train_loss = tr_loss / max(local_steps, 1)
        writer.add_scalar("train_loss", avg_train_loss, global_step)

        # 作者式宽松流程：每轮都看 valid 和 test
        valid_results = evaluate(model, eval_data_loader, args)
        test_results = inference(model, test_data_loader, args, save=False)

        for key, value in valid_results.items():
            writer.add_scalar(f"eval_{key}", value, global_step)

        for key, value in test_results.items():
            writer.add_scalar(f"test_{key}", value, global_step)
            logger.info(f"Epoch {epoch + 1} Test Set: {key}, {value}")

        writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("loss", avg_train_loss, global_step)

        # 按作者版：用 valid auc 选 best checkpoint
        if valid_results['auc'] > best_score:
            patience = 0
            best_score = valid_results['auc']
            torch.save(model.state_dict(), args.output_dir + 'valid_best_checkpoint.pth')
        else:
            patience += 1
            if patience == args.patience:
                break

    writer.flush()
    writer.close()
    return global_step, avg_train_loss


def evaluate(model, data_loader, args):
    criterion = get_loss_function()
    total_preds = []
    total_labels = []
    drug_gate_values = []
    target_gate_values = []

    # collect gate weights only during test inference
    drug_gate_values = []
    target_gate_values = []

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _, data in tqdm(
            enumerate(BackgroundGenerator(data_loader)),
            total=len(data_loader),
            desc="evaluate"
        ):
            compounds, atom, node, edge, attr, proteins, compound_mask, protein_mask, labels = [
                d.cuda() for d in data
            ]

            predicts = model.forward(
                compounds, proteins, atom, node, edge, attr, compound_mask, protein_mask
            )
            loss = criterion(predicts, labels.long())
            total_loss += loss.item()

            total_preds.append(predicts.cpu())
            total_labels.append(labels.cpu())

    total_preds = torch.cat(total_preds, 0).numpy()
    total_labels = torch.cat(total_labels, 0).numpy()

    preds_classes = np.argmax(total_preds, axis=1)

    # 按作者版：AUC/AUPR 使用 raw logits[:, 1]
    acc_valid = accuracy_score(total_labels, preds_classes)
    auc_valid = roc_auc_score(total_labels, total_preds[:, 1])
    aucpr_valid = average_precision_score(total_labels, total_preds[:, 1])

    precision_valid = precision_score(total_labels, preds_classes, zero_division=0)
    recall_valid = recall_score(total_labels, preds_classes, zero_division=0)

    return {
        'loss': total_loss / len(data_loader),
        'acc': acc_valid,
        'auc': auc_valid,
        'aucpr': aucpr_valid,
        'precision': precision_valid,
        'recall': recall_valid
    }


def inference(model, data_loader, args, state="Train", save=True):
    criterion = get_loss_function()
    model.eval()

    total_loss = 0.0
    total_preds = []
    total_labels = []
    drug_gate_values = []
    target_gate_values = []

    with torch.no_grad():
        for _, data in tqdm(
            enumerate(BackgroundGenerator(data_loader)),
            total=len(data_loader),
            desc='test'
        ):
            compounds, atom, node, edge, attr, proteins, compound_mask, protein_mask, labels = [
                d.cuda() for d in data
            ]

            predicts = model.forward(
                compounds, proteins, atom, node, edge, attr, compound_mask, protein_mask
            )
            loss = criterion(predicts, labels.long())
            if str(state).lower() == "test" and hasattr(model, "last_drug_gate"):
                dg = model.last_drug_gate
                tg = model.last_target_gate
                if dg.dim() == 3:
                    dg = dg.mean(dim=-1)
                if tg.dim() == 3:
                    tg = tg.mean(dim=-1)
                drug_gate_values.append(dg.reshape(-1).numpy())
                target_gate_values.append(tg.reshape(-1).numpy())

            total_loss += loss.item()

            total_preds.append(predicts.cpu())
            total_labels.append(labels.cpu())

    if str(state).lower() == "test" and len(drug_gate_values) > 0:
        drug_gate_values = np.concatenate(drug_gate_values)
        target_gate_values = np.concatenate(target_gate_values)
        gate_path = os.path.join(args.output_dir, "gate_distribution_test.npz")
        np.savez(gate_path, drug_gate=drug_gate_values, target_gate=target_gate_values)
        print("saved gate distribution:", gate_path)
        print("Drug-side gate:", drug_gate_values.mean(), drug_gate_values.std())
        print("Target-side gate:", target_gate_values.mean(), target_gate_values.std())

    total_preds = torch.cat(total_preds, 0).numpy()
    total_labels = torch.cat(total_labels, 0).numpy()

    preds_classes = np.argmax(total_preds, axis=1)

    # 按作者版：AUC/AUPR 使用 raw logits[:, 1]
    acc_test = accuracy_score(total_labels, preds_classes)
    auc_test = roc_auc_score(total_labels, total_preds[:, 1])
    aucpr_test = average_precision_score(total_labels, total_preds[:, 1])
    precision = precision_score(total_labels, preds_classes, zero_division=0)
    recall = recall_score(total_labels, preds_classes, zero_division=0)

    if save:
        with open(os.path.join(args.save_path, f"{args.dataset}_stable_{state}_prediction.txt"), 'a') as f:
            for label, pred in zip(total_labels, total_preds):
                f.write(f"{label} {pred}\n")

    return {
        'loss': total_loss / len(data_loader),
        'acc': acc_test,
        'auc': auc_test,
        'aucpr': aucpr_test,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Bank', help='dataset name: Davis, Metz, KIBA')
    parser.add_argument('--data_path', type=str, default='./datasets/', help='data path')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=500, help='warmup steps')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    parser.add_argument('--train_batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--head_num', type=int, default=2, help='head num')
    parser.add_argument('--conv', type=int, default=64, help='conv')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs to train for')
    parser.add_argument('--patience', type=int, default=25, help='early stopping patience')
    parser.add_argument('--k_fold', type=int, default=5, help='number of folds')
    parser.add_argument('--seed', type=int, default=4321, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--save_path', type=str, default='./Results/', help='save path')
    parser.add_argument('--log_path', type=str, default='./Results/', help='log path')
    parser.add_argument('--log_file', type=str, default='log.txt', help='log file')
    parser.add_argument('--model', type=str, default='AttentionDTA', help='model name')

    # 保留你现有扩展参数，方便继续跑你自己的模型
    parser.add_argument('--tf_layers', type=int, default=2)
    parser.add_argument('--tf_heads', type=int, default=4)
    parser.add_argument('--tf_ffn', type=int, default=256)
    parser.add_argument('--tf_dropout', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 保留参数位，但作者式协议版训练中不使用

    args = parser.parse_args()

    args.save_path = f'Results/{args.dataset}/lr_{args.lr}/size_{args.train_batch_size}/h1_{args.head_num}/conv_{args.conv}'
    os.makedirs(args.save_path, exist_ok=True)

    file_handler = logging.FileHandler(f'{args.save_path}/results.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(args)
    set_seed(args)

    logger.info("Train in {}".format(args.dataset))
    with open(os.path.join(args.data_path, f"{args.dataset}.txt"), 'r') as f:
        cpi_list = f.read().strip().split('\n')

    logger.info("load finished")
    logger.info("data shuffle")
    dataset = shuffle_dataset(cpi_list, args.seed)

    acc_List, auc_List, aupr_List, precision_List, recall_List = [], [], [], [], []

    for i_fold in range(args.k_fold):
        args.output_dir = f"{args.save_path}/{i_fold + 1}_Fold/"
        logger.info('*' * 25)
        logger.info('第' + str(i_fold + 1) + '折')
        logger.info('*' * 25)

        trainset, testset = get_kfold_data(i_fold, dataset, k=args.k_fold)

        TVdataset = CustomDataSet(trainset)
        test_dataset = CustomDataSet(testset)

        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size

        # 按作者版：random_split，不额外固定 generator
        train_dataset, valid_dataset = torch.utils.data.random_split(
            TVdataset, [train_size, valid_size]
        )

        train_dataset_load = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        valid_dataset_load = DataLoader(
            valid_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        test_dataset_load = DataLoader(
            test_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )

        model = AttentionDTA(
            head_num=args.head_num,
            conv=args.conv,
            dropout_rate=args.dropout_rate,
            tf_layers=args.tf_layers,
            tf_heads=args.tf_heads,
            tf_ffn=args.tf_ffn,
            tf_dropout=args.tf_dropout
        ).cuda()

        global_step, tr_loss = train(
            model,
            train_dataset_load,
            valid_dataset_load,
            test_dataset_load,
            args
        )
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

        torch.save(model.state_dict(), args.output_dir + 'stable_checkpoint.pth')
        model.load_state_dict(torch.load(args.output_dir + "valid_best_checkpoint.pth"))

        testset_test_results = inference(model, test_dataset_load, args, state='Test')

        logger.info("results on {}th fold\n".format(i_fold + 1))
        logger.info("Test: " + str(testset_test_results))

        acc_List.append(testset_test_results['acc'])
        auc_List.append(testset_test_results['auc'])
        aupr_List.append(testset_test_results['aucpr'])
        precision_List.append(testset_test_results['precision'])
        recall_List.append(testset_test_results['recall'])

    logger.info('The results on {}:'.format(args.dataset))
    logger.info('acc(std):{:.4f}({:.4f})'.format(np.mean(acc_List), np.var(acc_List)))
    logger.info('auc(std):{:.4f}({:.4f})'.format(np.mean(auc_List), np.var(auc_List)))
    logger.info('aupr(std):{:.4f}({:.4f})'.format(np.mean(aupr_List), np.var(aupr_List)))
    logger.info('precision(std):{:.4f}({:.4f})'.format(np.mean(precision_List), np.var(precision_List)))
    logger.info('recall(std):{:.4f}({:.4f})'.format(np.mean(recall_List), np.var(recall_List)))
