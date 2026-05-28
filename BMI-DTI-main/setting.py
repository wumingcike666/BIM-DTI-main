# -*- coding: utf-8 -*-


import random
import os
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)

from dataset import CustomDataSet, collate_fn
from model import AttentionDTA
from prefetch_generator import BackgroundGenerator



formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
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


def get_data(dataset):
    drugs = []
    proteins = []
    for pair in dataset:
        pair = pair.strip().split()
        drugs.append(pair[0])
        proteins.append(pair[1])
    drugs = list(set(drugs))
    proteins = list(set(proteins))
    return drugs, proteins


def split_data(dataset, drugs, proteins):
    train, test_drug, test_protein, test_denovel = [], [], [], []
    for i in dataset:
        pair = i.strip().split()
        if pair[0] not in drugs and pair[1] not in proteins:
            train.append(i)
        elif pair[0] not in drugs and pair[1] in proteins:
            test_drug.append(i)
        elif pair[0] in drugs and pair[1] not in proteins:
            test_protein.append(i)
        elif pair[0] in drugs and pair[1] in proteins:
            test_denovel.append(i)
    return train, test_drug, test_protein, test_denovel


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def train(model, train_data_loader, eval_data_loader, test_data_loader, args):
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    criterion = get_loss_function()
    t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
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

        valid_results = evaluate(model, eval_data_loader, args)
        test_results = inference(model, test_data_loader, args)

        for key, value in valid_results.items():
            writer.add_scalar(f"eval_{key}", value, global_step)
        for key, value in test_results.items():
            writer.add_scalar(f"test_{key}", value, global_step)
            logger.info(f"Epoch {epoch + 1} Test Set: {key}, {value}")

        writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("loss", avg_train_loss, global_step)

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

    acc_valid = accuracy_score(total_labels, preds_classes)
    auc_valid = roc_auc_score(total_labels, total_preds[:, 1])
    aucpr_valid = average_precision_score(total_labels, total_preds[:, 1])

    return {
        'loss': total_loss / len(data_loader),
        'acc': acc_valid,
        'auc': auc_valid,
        'aucpr': aucpr_valid
    }


def inference(model, data_loader, args, state="Train", save=True):
    criterion = get_loss_function()
    model.eval()

    total_loss = 0.0
    total_preds = []
    total_labels = []

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
            total_loss += loss.item()

            total_preds.append(predicts.cpu())
            total_labels.append(labels.cpu())

    total_preds = torch.cat(total_preds, 0).numpy()
    total_labels = torch.cat(total_labels, 0).numpy()

    preds_classes = np.argmax(total_preds, axis=1)

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


def show_epoch_result(testset_test_results, precision_List, recall_List, acc_List, auc_List, aupr_List):
    logger.info("results on {}th fold\n".format(i_fold + 1))
    logger.info("Test: " + str(testset_test_results))

    acc_List.append(testset_test_results['acc'])
    auc_List.append(testset_test_results['auc'])
    aupr_List.append(testset_test_results['aucpr'])
    precision_List.append(testset_test_results['precision'])
    recall_List.append(testset_test_results['recall'])


def show_result(precision_List, recall_List, acc_List, auc_List, aupr_List):
    logger.info('acc(std):{:.4f}({:.4f})'.format(np.mean(acc_List), np.var(acc_List)))
    logger.info('auc(std):{:.4f}({:.4f})'.format(np.mean(auc_List), np.var(auc_List)))
    logger.info('aupr(std):{:.4f}({:.4f})'.format(np.mean(aupr_List), np.var(aupr_List)))
    logger.info('precision(std):{:.4f}({:.4f})'.format(np.mean(precision_List), np.var(precision_List)))
    logger.info('recall(std):{:.4f}({:.4f})'.format(np.mean(recall_List), np.var(recall_List)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Bank', help='dataset name')
    parser.add_argument('--data_path', type=str, default='./datasets/', help='data path')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=500, help='warmup steps')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')

    parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
    parser.add_argument('--head_num', type=int, default=4, help='head num')

    parser.add_argument('--conv', type=int, default=64, help='cnn hidden channels')
    parser.add_argument('--tf_layers', type=int, default=1, help='transformer layers')
    parser.add_argument('--tf_heads', type=int, default=4, help='transformer heads')
    parser.add_argument('--tf_ffn', type=int, default=192, help='transformer ffn dim')
    parser.add_argument('--tf_dropout', type=float, default=0.10, help='transformer dropout')
    parser.add_argument('--dropout_rate', type=float, default=0.10, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--epochs', type=int, default=120, help='the number of epochs to train for')
    parser.add_argument('--patience', type=int, default=20, help='early stop patience')
    parser.add_argument('--k_fold', type=int, default=5, help='number of folds')
    parser.add_argument('--seed', type=int, default=4321, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')

    parser.add_argument('--save_path', type=str, default='./Results/', help='save path')
    parser.add_argument('--log_path', type=str, default='./Results/', help='log path')
    parser.add_argument('--log_file', type=str, default='log.txt', help='log file')
    parser.add_argument('--model', type=str, default='AttentionDTA', help='model name')

    args = parser.parse_args()

    args.save_path = f'Results/{args.dataset}/lr_{args.lr}'
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
    drugs, proteins = get_data(dataset)

    Precision_List_stable_drug, Recall_List_stable_drug, Accuracy_List_stable_drug, AUC_List_stable_drug, AUPR_List_stable_drug = [], [], [], [], []
    Precision_List_stable_protein, Recall_List_stable_protein, Accuracy_List_stable_protein, AUC_List_stable_protein, AUPR_List_stable_protein = [], [], [], [], []
    Precision_List_stable_deno, Recall_List_stable_deno, Accuracy_List_stable_deno, AUC_List_stable_deno, AUPR_List_stable_deno = [], [], [], [], []

    for i_fold in range(args.k_fold):
        args.output_dir = f"{args.save_path}/{i_fold + 1}_Fold/"

        logger.info('*' * 25)
        logger.info('第' + str(i_fold + 1) + '折')
        logger.info('*' * 25)

        _, test_drugs = get_kfold_data(i_fold, drugs, k=args.k_fold)
        _, test_proteins = get_kfold_data(i_fold, proteins, k=args.k_fold)

        train_dataset, test_dataset_drug, test_dataset_protein, test_dataset_denovel = split_data(
            dataset, test_drugs, test_proteins
        )

        TVdataset = CustomDataSet(train_dataset)
        test_dataset_drug = CustomDataSet(test_dataset_drug)
        test_dataset_protein = CustomDataSet(test_dataset_protein)
        test_dataset_denovel = CustomDataSet(test_dataset_denovel)

        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size

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
        test_dataset_drug_load = DataLoader(
            test_dataset_drug,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        test_dataset_protein_load = DataLoader(
            test_dataset_protein,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        test_dataset_denovel_load = DataLoader(
            test_dataset_denovel,
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
            test_dataset_drug_load,
            args
        )
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

        torch.save(model.state_dict(), args.output_dir + 'stable_checkpoint.pth')
        model.load_state_dict(torch.load(args.output_dir + "valid_best_checkpoint.pth"))

        testset_test_drug_results = inference(model, test_dataset_drug_load, args, state='Test')
        show_epoch_result(
            testset_test_drug_results,
            Precision_List_stable_drug, Recall_List_stable_drug, Accuracy_List_stable_drug,
            AUC_List_stable_drug, AUPR_List_stable_drug
        )

        testset_test_protein_results = inference(model, test_dataset_protein_load, args, state='Test')
        show_epoch_result(
            testset_test_protein_results,
            Precision_List_stable_protein, Recall_List_stable_protein, Accuracy_List_stable_protein,
            AUC_List_stable_protein, AUPR_List_stable_protein
        )

        testset_test_pair_results = inference(model, test_dataset_denovel_load, args, state='Test')
        show_epoch_result(
            testset_test_pair_results,
            Precision_List_stable_deno, Recall_List_stable_deno, Accuracy_List_stable_deno,
            AUC_List_stable_deno, AUPR_List_stable_deno
        )

        if i_fold == 2:
            break

    show_result(
        Precision_List_stable_drug, Recall_List_stable_drug, Accuracy_List_stable_drug,
        AUC_List_stable_drug, AUPR_List_stable_drug
    )
    show_result(
        Precision_List_stable_protein, Recall_List_stable_protein, Accuracy_List_stable_protein,
        AUC_List_stable_protein, AUPR_List_stable_protein
    )
    show_result(
        Precision_List_stable_deno, Recall_List_stable_deno, Accuracy_List_stable_deno,
        AUC_List_stable_deno, AUPR_List_stable_deno
    )