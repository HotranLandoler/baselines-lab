import argparse

import torch
import torch.nn.functional as func
import torch_geometric
from torch import autograd, Tensor
from torch_geometric.data import Data
from torch_sparse import SparseTensor

import models.graph_smote
import options
import utils
from logger import Logger
from evaluator import Evaluator
from early_stopping import EarlyStopping


encoder = models.graph_smote.SageEncoder(17, 32, 32, 0.0).to("cuda:0")
decoder = models.graph_smote.Decoder(32, 0.0).to("cuda:0")

def main():
    # Experiment setup
    args = options.prepare_args()
    utils.set_random_seed(args.random_seed)

    # Prepare data and model
    data, model = utils.prepare_data_and_model(args)

    weight = utils.get_loss_weight(args)
    evaluator = Evaluator(args.metrics, num_classes=args.num_classes)
    logger = Logger(settings=args)

    # print(model)
    # for name, p in model.named_parameters():
    #     print(name, ' :', p.shape)
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

    if args.model == "amnet":
        optimizer = torch.optim.Adam([
            dict(params=model.filters.parameters(), lr=5e-2),
            dict(params=model.lin, lr=args.lr, weight_decay=args.weight_decay),
            dict(params=model.attn, lr=args.lr, weight_decay=args.weight_decay)]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    # For AMNet
    train_mask_bool = torch.zeros(data.num_nodes, dtype=torch.bool, device=args.device)
    train_mask_bool[data.train_mask] = True
    anomaly_label = train_mask_bool & (data.y == 1)
    normal_label = train_mask_bool & (data.y == 0)

    for epoch in range(args.epochs):
        train_loss = _train_epoch(model, data, edge_index,
                                  optimizer, args, loss_weight=loss_weight,
                                  anomaly_label=anomaly_label,
                                  normal_label=normal_label)
        val_loss = _validate_epoch(model, data, edge_index, args,
                                   loss_weight=loss_weight)
        print(f"Epoch {epoch} finished. "
              f"train_loss: {train_loss:>7f} "
              f"val_loss: {val_loss:>7f}")
        if args.plot:
            logger.add_record(train_loss, val_loss)

        if (args.use_early_stopping and
                early_stopping.check_stop(val_loss)):
            print("Early Stopping")
            total_epochs = epoch + 1
            break

    # Test
    model.eval()
    predicts = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
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
                 loss_weight: torch.Tensor | None,
                 anomaly_label: Tensor,
                 normal_label: Tensor) -> float:
    model.train()
    optimizer.zero_grad()

    # x, y_new, train_mask_new = Smote._recon_upsample(data.x, data.y, data.train_mask)

    # with autograd.detect_anomaly():
    if args.model == "amnet":
        output, bias_loss = model(data.x, edge_index,
                                  label=(anomaly_label, normal_label))
        beta = 1.0
        loss = (func.nll_loss(output[data.train_mask], data.y[data.train_mask]) +
                bias_loss * beta)
    else:
        h = encoder(data.x, edge_index)
        adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index)
        h, y_new, train_mask_new, adj_up = models.graph_smote.GraphSmote.recon_upsample(h, data.y, data.train_mask,
                                                                                        adj=adj,
                                                                                        portion=1.0)
        generated_G = decoder(h)
        # output, label_scores = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
        output = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
        loss = func.nll_loss(output[data.train_mask], data.y[data.train_mask],
                             weight=loss_weight)
        # label_loss_factor = 0.0
        # loss += label_loss_factor * func.cross_entropy(label_scores[data.train_mask], data.y[data.train_mask])

    loss.backward()

    optimizer.step()
    # print(f"Epoch {epoch} finished. loss: {loss.item():>7f}")
    return loss.item()


def _validate_epoch(model: torch.nn.Module,
                    data: Data,
                    edge_index: Tensor | SparseTensor,
                    args: argparse.Namespace,
                    loss_weight: torch.Tensor | None) -> float:
    model.eval()
    predicts = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
    val_loss = func.nll_loss(predicts[data.val_mask], data.y[data.val_mask],
                             weight=loss_weight)
    return val_loss.item()


def _model_wrapper(model: torch.nn.Module,
                   x: Tensor,
                   edge_index: Tensor | SparseTensor,
                   data: Data,
                   drop_rate: float) -> Tensor:
    return model(x, edge_index, data=data, drop_rate=drop_rate)


if __name__ == "__main__":
    main()
