import argparse
import multiprocessing as mp
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator
import torch.nn.functional as F
# noinspection PyUnresolvedReferences
from data import SubgraphData
from utils import get_data, get_model, SimpleEvaluator, NonBinaryEvaluator, Evaluator

torch.set_num_threads(1)

def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def cross_entropy(logits, labels, reduction='mean'):
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def adjust_lr_beta1(optimizer, lr, beta1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, 0.999)  # Only change beta1


def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def train(model, device, loader, optimizer, criterion, epoch, fold_idx):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, pred_preturb = model(batch)
            optimizer.zero_grad()
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

            y = batch.y.view(pred.shape).to(torch.float32) if pred.size(-1) == 1 else batch.y
            # loss = criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])
            loss = criterion(pred, batch.y)

            # wandb.log({f'Loss/train': loss.item()})
            loss.backward()
            optimizer.step()

def evaluate( model, device, dataloader, evaluator, given_labels):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            logs, logs_preturb = model(data)
            logs_soft = F.log_softmax(logs, dim=-1)
            pred = logs_soft.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.cross_entropy(logs, data.y) * data.num_graphs
    test_accuracy = correct / len(dataloader.dataset)
    # print('test acc:', test_accuracy)
    return test_accuracy, loss/len(dataloader.dataset)

def eval(model, device, loader, evaluator):
    model.eval()

    all_y_pred = []
    # for i in range(voting_times):
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _ = model(batch)

            y = batch.y.view(pred.shape) if pred.size(-1) == 1 else batch.y
            y_true.append(y.detach().cpu())
            y_pred.append(pred.detach().cpu())

    all_y_pred.append(torch.cat(y_pred, dim=0).unsqueeze(-1).numpy())

    y_true = torch.cat(y_true, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": all_y_pred}
    return evaluator.eval(input_dict)


def step_flagging(content):
    print('\n=================================================')
    print(content, flush=True)
    print('=================================================')

def get_loader(dataset, split_idx, args):

    train_loader = DataLoader(dataset[split_idx["train"]],
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, follow_batch=['subgraph_idx'])
    train_loader_eval = DataLoader(dataset[split_idx["train"]],
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, follow_batch=['subgraph_idx'])
    valid_loader = DataLoader(dataset[split_idx["valid"]],
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, follow_batch=['subgraph_idx'])
    test_loader = DataLoader(dataset[split_idx["test"]],
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, follow_batch=['subgraph_idx'])

    in_dim = dataset.num_features
    out_dim = dataset.num_tasks if args.dataset != 'ZINC' else 1

    task_type = dataset.task_type
    eval_metric = dataset.eval_metric
    return train_loader, train_loader_eval, valid_loader, test_loader, (in_dim, out_dim, task_type, eval_metric)


class EMA(object):
    """
    Usage:
        model = ResNet(config)
        ema = EMA(model, alpha=0.999)
        ... # train an epoch
        ema.update_params(model)
        ema.apply_shadow(model)
    """
    def __init__(self, model, alpha=0.999):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]
        self.alpha = alpha

    def init_params(self, model):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]

    def update_params(self, model):
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(self.alpha * self.shadow[name] + (1 - self.alpha) * state[name])

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow, strict=True)

def run(args, device, fold_idx):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset, split_idx = get_data(args, fold_idx)

    class Add_Noise_Binary(object):
        def __call__(self, data):
            data.y = torch.Tensor([1]).long() - data.y
            return data

    class Add_Noise_Multi(object):
        def __call__(self, data):
            label_list = list(range(dataset.num_classes))
            label_list.remove(data.y.item())
            selected = random.choice(label_list)
            data.y = torch.Tensor([selected]).long()
            return data

    def add_noise(train_idx):
        sample = int(args.noise_ratio * len(train_idx))
        # print(sample)
        sample_idx = random.sample(list(train_idx), sample)
        # print('idx:', len(sample_idx),len(dataset))
        data_list = []
        for i in range(len(dataset)):
            data_list.append(dataset.get(i))
        # if args.dataset in ['MUTAG', 'PTC', 'PROTEINS', 'IMDB-BINARY', 'NCI1', 'REDDIT-BINARY', 'DD']:
        data_list_fin = []
        for i, data in enumerate(data_list):
            if i not in sample_idx:
                data_list_fin.append(data)
            else:
                data_list_fin.append(Add_Noise_Binary()(data))
        # else:
        #     data_list_fin = []
        #     for i, data in enumerate(data_list):
        #         if i not in sample_idx:
        #             data_list_fin.append(data)
        #         else:
        #             data_list_fin.append(Add_Noise_Multi()(data))

        noise_dataset = deepcopy(dataset)
        noise_dataset.__indices__ = None
        noise_dataset.__data_list__ = data_list
        noise_dataset.data, noise_dataset.slices = dataset.collate(data_list_fin)
        # print(len(sample_idx))
        # print(dataset.data.y[sample_idx])
        # print(noise_dataset.data.y[sample_idx])

        return noise_dataset

    noise_dataset = add_noise(split_idx['train'])

    train_loader, train_loader_eval, valid_loader, test_loader, attributes = get_loader(noise_dataset, split_idx, args)

    in_dim, out_dim, task_type, eval_metric = attributes

    if 'ogb' in args.dataset:
        evaluator = Evaluator(args.dataset)
    else:
        evaluator = SimpleEvaluator(task_type) if args.dataset != "IMDB-MULTI" \
                                                  and args.dataset != "CSL" else NonBinaryEvaluator(out_dim)

    model = get_model(args, in_dim, out_dim, device, args.sample)
    model_ema = get_model(args, in_dim, out_dim, device, args.sample)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    # if "classification" in task_type:
    # criterion = torch.nn.BCEWithLogitsLoss() if args.dataset != "IMDB-MULTI" \
    #                                                 and args.dataset != "COLLAB" else torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # else:
    #     criterion = torch.nn.L1Loss()

    # If sampling, perform majority voting on the outputs of 5 independent samples
    # voting_times = 1

    train_curve = []
    valid_curve = []
    test_curve = []

    flag = 0

    best_val = 0
    test_acc = 0

    patience = 0

    ema = EMA(model, alpha=0.999)
    ema.apply_shadow(model_ema)

    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(1, args.epochs + 1)):

        # train(model, device, train_loader, optimizer, criterion, epoch=epoch, fold_idx=fold_idx)
        ''''''

        model.train()
        # net_ema.train()

        if epoch < args.warmup_epochs:
            threshold_clean = min(args.tau_clean * epoch / args.warmup_epochs, args.tau_clean)
        correct_train = 0
        train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            logs, aux_logs = model(data)
            logs_ema, aux_logs_ema = model_ema(data)
            prob1 = F.log_softmax(logs, dim=-1)

            N, C = logs.shape
            given_labels = torch.full(size=(N, C), fill_value=args.eps / (C - 1)).to(device)
            # given_labels = torch.full(size=(N, C), fill_value=0).to(device)
            given_labels.scatter_(dim=1, index=torch.unsqueeze(data.y, dim=1), value=1 - args.eps)
            # given_labels.scatter_(dim=1, index=torch.unsqueeze(data.y, dim=1), value=1)
            with torch.no_grad():
                subgraph_list = []
                subgraph_list_ema = []
                for subgraph in aux_logs:
                    subgraph_list.append(torch.mean(subgraph, dim=0))
                subgraphs = torch.stack(subgraph_list, dim=0)  #32,2
                for subgraph in aux_logs_ema:
                    subgraph_list_ema.append(torch.mean(subgraph, dim=0))
                subgraphs_ema = torch.stack(subgraph_list_ema, dim=0)  #32,2]
                prob2 = F.log_softmax(subgraphs, dim=-1)
                # print(subgraphs.shape)
                soft_labels = (F.log_softmax(subgraphs_ema, dim=-1) + F.log_softmax(logs_ema, dim=-1)) / 2
                # soft_labels_logs = F.log_softmax(logs, dim=-1)
                soft_1 = F.softmax(subgraphs, dim=-1)
                soft_2 = F.softmax(logs, dim=-1)
                prob_clean = 2 - js_div(soft_1, given_labels) - js_div(soft_2, given_labels)
                # prob_clean = 1 - js_div(soft_1, given_labels)
                prob_clean = torch.where(torch.isnan(prob_clean), torch.full_like(prob_clean, 0), prob_clean)

            if epoch < args.warmup_epochs:
                if flag == 0:
                    step_flagging(f'start the warm-up step for {args.warmup_epochs} epochs.')
                    flag += 1
                losses = cross_entropy(logs, given_labels, reduction='none')
                loss = losses[prob_clean >= threshold_clean].mean()

            else:
                if flag == 1:
                    step_flagging('start the robust learning step.')
                    flag += 1

                target_labels = given_labels.clone()
                # clean samples
                idx_clean = (prob_clean >= threshold_clean).nonzero(as_tuple=False).squeeze(dim=1)
                _, preds1 = prob1.topk(1, 1, True, True)
                _, preds2 = prob2.topk(1, 1, True, True)

                disagree = (preds1 != preds2).squeeze(dim=1)
                agree = (preds1 == preds2).squeeze(dim=1)
                print(disagree, agree)
                unclean = (prob_clean < threshold_clean)
                idx_id = (agree * unclean).nonzero(as_tuple=False).squeeze(dim=1)
                idx_ood = (disagree * unclean).nonzero(as_tuple=False).squeeze(dim=1)
                # idx_ood = unclean.nonzero(as_tuple=False).squeeze(dim=1)
                target_labels[idx_id] = soft_labels[idx_id]
                target_labels[idx_ood] = F.softmax(given_labels[idx_ood] / args.T, dim=1)

                losses = cross_entropy(logs, target_labels, reduction='none') #logs->subgraphs
                loss_c = losses.mean()

                # consistency loss
                sign = torch.ones(N).to(device)
                sign[idx_ood] *= -1

                losses_o = 0
                for idx, subgraph in enumerate(aux_logs):
                    losses_o += (logs[idx,:].repeat(subgraph.shape[0],1) - subgraph).norm(dim=-1).mean()

                loss_o = losses_o.mean()
                # print(loss_c, loss_o)

                # final loss
                loss = (1 - args.alpha) * loss_c + loss_o * args.alpha
                # print(loss)
                # loss = loss_c

            pred = prob1.max(1)[1]
            correct_train += pred.eq(data.y).sum().item()
            train_loss += loss * data.num_graphs
            loss.backward()

            optimizer.step()
            ema.update_params(model)
            ema.apply_shadow(model_ema)
        scheduler.step()

        train_loss_list.append(train_loss/len(train_loader.dataset))

        valid_perf, val_loss = evaluate(model, device, valid_loader, evaluator, given_labels)
        val_loss_list.append(val_loss)

        # if best_val < valid_perf:
        #     best_val = valid_perf
            # test_acc, _ = evaluate(model, device, test_loader, evaluator, given_labels)



        if scheduler is not None:
            if 'ZINC' in args.dataset:
                scheduler.step(valid_perf[eval_metric])
                if optimizer.param_groups[0]['lr'] < 0.00001:
                    break
            else:
                scheduler.step()

        valid_curve.append(valid_perf)
        print(f"Epoch:[{epoch + 1:>3d}/{args.epochs:>3d}],  "
              f"Test Accuracy:[{valid_perf:6.2f}],  "
              )

        # if best_val > valid_perf:
        #     patience += 1
        #     if patience >= args.patience:
        #         return valid_curve, train_loss_list, val_loss_list

    return valid_curve, train_loss_list, val_loss_list


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str, default='gin',
                        help='Type of convolution {gin, originalgin, zincgin, graphconv}')
    parser.add_argument('--random_ratio', type=float, default=0.,
                        help='Number of random features, > 0 only for RNI')
    parser.add_argument('--model', type=str, default='deepsets',
                        help='Type of model {deepsets, dss, gnn}')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--channels', type=str, default='64-64',
                        help='String with dimension of each DS layer, separated by "-"'
                             '(considered only if args.model is deepsets)')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--jk', type=str, default="last",
                        help='JK strategy, either last or concat (default: last)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training (default: 0.01)')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay rate for training (default: 0.5)') #warmup_epochs
    parser.add_argument('--tau_clean', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--decay_step', type=int, default=50,
                        help='decay step for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--noise_ratio', type=float, default=0.8,
                        )
    parser.add_argument('--policy', type=str, default="node_deleted",
                        help='Subgraph selection policy in {edge_deleted, node_deleted, ego_nets}'
                             ' (default: edge_deleted)')
    parser.add_argument('--num_hops', type=int, default=2,
                        help='Depth of the ego net if policy is ego_nets (default: 2)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of subsampled subgraphs (1.0 means full bag aka no sampling)')
    parser.add_argument('--patience', type=int, default=20,
                        help='patience (default: 20)')
    parser.add_argument('--test', action='store_true',
                        help='quick test')

    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')

    args = parser.parse_args()


    if args.dataset == 'PROTEINS' or args.dataset == 'PROTEINS_full':
        args.emb_dim = 16
    print(args)
    args.channels = list(map(int, args.channels.split("-")))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    mp.set_start_method('spawn')

    n_folds = 10

    # number of processes to run in parallel
    # TODO: make it dynamic
    if n_folds > 1 and 'REDDIT' not in args.dataset:
        if args.dataset == 'PROTEINS':
            num_proc = 2
        else:
            num_proc = 3 if args.batch_size == 128 and args.dataset != 'MUTAG' and args.dataset != 'PTC' else 5
    else:
        num_proc = 1

    if args.dataset in ['CEXP', 'EXP']:
        num_proc = 2
    if 'IMDB' in args.dataset and args.policy == 'edge_deleted':
        num_proc = 1

    num_free = 1
    # results_queue = mp.Queue()

    curve_folds_train = []
    curve_folds_val = []
    curve_folds_test = []
    fold_idx = 0

    if args.test:
        # run(args, device, fold_idx, sweep_run_name, sweep_id, results_queue)
        run(args, device, fold_idx)
        exit()

    train_loss_total = []
    val_loss_total = []
    while len(curve_folds_val) < n_folds:
        valid_curve, train_loss_list, val_loss_list = run(args, device, fold_idx)
        train_loss_total.append(torch.tensor(train_loss_list))
        val_loss_total.append(torch.tensor(val_loss_list))
        fold_idx += 1
        curve_folds_val.append(valid_curve)
        torch.cuda.empty_cache()

    # min_length = args.epochs
    # for leg in curve_folds_val:
    #     # print(len(leg))
    #     if min_length > len(leg):
    #         # print(min_length, len(leg))
    #         min_length = len(leg)

    # train_loss = torch.stack(train_loss_total, dim=0).mean(dim=0)
    # val_loss = torch.stack(val_loss_total, dim=0).mean(dim=0)
    valid_curve_folds = np.array([l for l in curve_folds_val])

    valid_curve = np.mean(valid_curve_folds, 0)
    valid_accs_std = np.std(valid_curve_folds, 0)

    # task_type = 'classification' if args.dataset != 'ZINC' else 'regression'
    # if 'classification' in task_type:
    best_val_epoch = np.argmax(valid_curve)
    print('best epoch:', best_val_epoch)
    print('Test:' + str(valid_curve[best_val_epoch]) + ' Test std:' + str(valid_accs_std[best_val_epoch]))

    # print(train_loss, val_loss)

    # wandb.join()


if __name__ == "__main__":
    main()
