import argparse

import torch
import torch.nn.functional as func
from torch import autograd, Tensor
from torch_geometric.data import Data
from torch_sparse import SparseTensor

import options
import utils
from logger import Logger
from evaluator import Evaluator
from early_stopping import EarlyStopping


def main():
    # Experiment setup
    args = options.prepare_args()
    utils.set_random_seed(args.random_seed)

    # Prepare data and model
    data, model = utils.prepare_data_and_model(args)

    weight = utils.get_loss_weight(args)
    evaluator = Evaluator(args.metrics, num_classes=args.num_classes)
    logger = Logger(settings=args)

    print(model)
    print(f"Train started with setting {args}")

    edge_index = data.adj_t if hasattr(data, "adj_t") else data.edge_index

    total_epochs: int = args.epochs
    for run in range(args.runs):
        total_epochs = _train_run(run, model, data, edge_index,
                                  args, evaluator, logger,
                                  loss_weight=weight)

    print("Train ended.")
    for result in evaluator.get_final_results():
        print(result, end="")
    print("")

    if args.save_log:
        logger.save_to_file(evaluator, path=args.log_path)

    if args.plot:
        if args.runs > 1:
            raise NotImplementedError("Loss plot not implemented for multiple runs")
        logger.plot_and_save(total_epochs)


def _train_run(run: int,
               model: torch.nn.Module,
               data: Data,
               edge_index: Tensor | SparseTensor,
               args: argparse.Namespace,
               evaluator: Evaluator,
               logger: Logger,
               loss_weight: torch.Tensor) -> int:
    print(f"Run {run} " + "-" * 20)
    # gc.collect()
    total_epochs = args.epochs

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    for epoch in range(args.epochs):
        train_loss = _train_epoch(model, data, edge_index,
                                  optimizer, args, loss_weight=loss_weight)
        val_loss = _validate_epoch(model, data, edge_index,
                                   loss_weight=loss_weight)
        print(f"Epoch {epoch} finished. "
              f"train_loss: {train_loss:>7f} "
              f"val_loss: {val_loss:>7f}")
        logger.add_record(train_loss, val_loss)

        if (args.use_early_stopping and
                early_stopping.check_stop(val_loss)):
            print("Early Stopping")
            total_epochs = epoch + 1
            break

    # Test
    model.eval()
    predicts = model(data.x, edge_index, data=data)
    # result = metric(predicts[dataset.test_mask], dataset.y[dataset.test_mask], num_classes=OUT_FEATS)
    # logger.add_result(result.item())
    print(f"Run {run}: ", end='')
    for result in evaluator.evaluate_test(predicts, data.y, data.test_mask):
        print(result, end='')
    print("")

    return total_epochs


def _train_epoch(model: torch.nn.Module,
                 data: Data,
                 edge_index: Tensor | SparseTensor,
                 optimizer: torch.optim.Optimizer,
                 args: argparse.Namespace,
                 loss_weight: torch.Tensor | None) -> float:
    model.train()
    optimizer.zero_grad()

    # with autograd.detect_anomaly():
    if args.model == "dropgcn":
        output = model(data.x, edge_index, drop_rate=args.drop_rate)
    else:
        output = model(data.x, edge_index, data=data)

    loss = func.nll_loss(output[data.train_mask], data.y[data.train_mask],
                         weight=loss_weight)
    loss.backward()

    optimizer.step()
    # print(f"Epoch {epoch} finished. loss: {loss.item():>7f}")
    return loss.item()


def _validate_epoch(model: torch.nn.Module,
                    data: Data,
                    edge_index: Tensor | SparseTensor,
                    loss_weight: torch.Tensor | None) -> float:
    model.eval()
    predicts = model(data.x, edge_index, data=data)
    val_loss = func.nll_loss(predicts[data.val_mask], data.y[data.val_mask],
                             weight=loss_weight)
    return val_loss.item()


if __name__ == "__main__":
    main()
