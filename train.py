import argparse
import typing

import torch
import torch.nn.functional as func
import torch_geometric
from torch import autograd, Tensor
from torch_geometric.data import Data
from torch_sparse import SparseTensor

import options
import utils
from logger import Logger
from evaluator import Evaluator
from early_stopping import EarlyStopping
from models.graph_smote import Classifier
from models.dagad import GeneralizedCELoss1


def main():
    # torch.autograd.set_detect_anomaly(True)
    # Experiment setup
    args = options.prepare_args()
    utils.set_random_seed(args.random_seed)

    # Prepare data and model
    data, model = utils.prepare_data_and_model(args)
    # torch_geometric.compile(model)

    model_classifier = Classifier(args.hidden_size, args.num_classes).to(args.device)

    weight = utils.get_loss_weight(args)
    evaluator = Evaluator(args.metrics, num_classes=args.num_classes, runs=args.runs)
    logger = Logger(settings=args)

    # print(model)
    # for name, p in model.named_parameters():
    #     print(name, ' :', p.shape)
    print(f"Train started with setting {args}")

    edge_index = data.adj_t if hasattr(data, "adj_t") else data.edge_index

    total_epochs: int = args.epochs
    for run in range(args.runs):
        total_epochs = _train_run(run, model, model_classifier, data, edge_index,
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
               model_classifier: torch.nn.Module,
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

    optimizer_classifier = torch.optim.Adam(model_classifier.parameters(), lr=args.lr,
                                            weight_decay=args.weight_decay)

    criterion_gce = GeneralizedCELoss1(q=0.7)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    val_loss_best = 99.
    for epoch in range(args.epochs):
        if epoch % 30 == 0:
            permute = True
        else:
            permute = False

        train_loss = _train_epoch(model, model_classifier, data, edge_index,
                                  optimizer, optimizer_classifier,
                                  args, criterion_gce, permute,
                                  loss_weight=loss_weight)
        out, val_loss = _validate_epoch(model, model_classifier, data, edge_index, args,
                                        loss_weight=loss_weight)

        # with torch.no_grad():
        #     evaluator.evaluate_test(run, out.exp(), data.y, data.test_mask)

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            with torch.no_grad():
                evaluator.evaluate_test(run, out.exp(), data.y, data.test_mask)

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
    with torch.no_grad():
        model.eval()
        # _, out, *_ = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
        # evaluator.evaluate_test(run, out.exp(), data.y, data.test_mask)
        evaluator.print_run_results(run)

        if args.visualize:
            # The first return value is the embedding
            embedding, *_ = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
            logger.plot_embedding_visualization(embedding[data.test_mask],
                                                data.y[data.test_mask])

    return total_epochs


def _train_epoch(model: torch.nn.Module,
                 model_classifier: torch.nn.Module,
                 data: Data,
                 edge_index: Tensor | SparseTensor,
                 optimizer: torch.optim.Optimizer,
                 optimizer_classifier: torch.optim.Optimizer,
                 args: argparse.Namespace,
                 criterion_gce: GeneralizedCELoss1,
                 permute: bool,
                 loss_weight: torch.Tensor | None) -> float:
    model.train()
    optimizer.zero_grad()
    optimizer_classifier.zero_grad()

    # with autograd.detect_anomaly():
    if args.model == "amnet":
        train_mask_bool = torch.zeros(data.num_nodes, dtype=torch.bool, device=args.device)
        train_mask_bool[data.train_mask] = True
        anomaly_label = train_mask_bool & (data.y == 1)
        normal_label = train_mask_bool & (data.y == 0)
        _, output, bias_loss = model(data.x, edge_index,
                                     label=(anomaly_label, normal_label))
        beta = 1.0
        loss = (func.nll_loss(output[data.train_mask], data.y[data.train_mask],
                              weight=loss_weight) +
                bias_loss * beta)
    elif args.model == "dagad":
        criterion = func.cross_entropy
        alpha = 1.5
        beta = 0.5
        pred_org_a, pred_org_b, _, pred_aug_bcak_b, data = model(data, permute=permute)

        loss_ce_a = criterion(pred_org_a[data.train_mask], data.y[data.train_mask])
        loss_ce_b = criterion(pred_org_b[data.train_mask], data.y[data.train_mask])
        loss_ce_weight = loss_ce_b / (loss_ce_b + loss_ce_a + 1e-8)
        loss_ce_anm = criterion(pred_org_a[data.train_anm], data.y[data.train_anm])
        loss_ce_norm = criterion(pred_org_a[data.train_norm], data.y[data.train_norm])
        loss_ce = loss_ce_weight * (loss_ce_anm + loss_ce_norm) / 2

        loss_gce = 0.5 * criterion_gce(pred_org_b[data.train_anm], data.y[data.train_anm]) \
                   + 0.5 * criterion_gce(pred_org_b[data.train_norm], data.y[data.train_norm])

        loss_gce_aug = 0.5 * criterion_gce(pred_aug_bcak_b[data.aug_train_anm], data.aug_y[data.aug_train_anm]) \
                       + 0.5 * criterion_gce(pred_aug_bcak_b[data.aug_train_norm], data.aug_y[data.aug_train_norm])

        loss = alpha * loss_ce + loss_gce + beta * loss_gce_aug
    elif args.model == "gfca":
        _, output, y_new, train_mask_new = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
        # embedding, y_new, train_mask_new = GraphSmote.recon_upsample(embedding, data.y, data.train_mask)
        # output = model_classifier(embedding)
        loss = func.nll_loss(output[train_mask_new], y_new[train_mask_new],
                             weight=loss_weight)
    else:
        # output, label_scores = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
        _, output = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
        loss = func.nll_loss(output[data.train_mask], data.y[data.train_mask],
                             weight=loss_weight)
        # label_loss_factor = 0.0
        # loss += label_loss_factor * func.cross_entropy(label_scores[data.train_mask], data.y[data.train_mask])

    loss.backward()

    optimizer.step()
    if args.model == "smote":
        optimizer_classifier.step()
    # print(f"Epoch {epoch} finished. loss: {loss.item():>7f}")
    return loss.item()


@torch.no_grad()
def _validate_epoch(model: torch.nn.Module,
                    model_classifier: torch.nn.Module,
                    data: Data,
                    edge_index: Tensor | SparseTensor,
                    args: argparse.Namespace,
                    loss_weight: torch.Tensor | None) -> (Tensor, float):
    model.eval()
    if args.model == "dagad":
        criterion = func.cross_entropy
        _, pred_org_b, _, _, data = model(data, permute=False)
        predicts = pred_org_b
    elif args.model == "gfca":
        criterion = func.nll_loss
        _, predicts, _, _ = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)
        # predicts = model_classifier(embedding)
    else:
        criterion = func.nll_loss
        _, predicts = _model_wrapper(model, data.x, edge_index, data, args.drop_rate)

    val_loss = criterion(predicts[data.val_mask], data.y[data.val_mask],
                         weight=loss_weight)
    return predicts, val_loss.item()


def _model_wrapper(model: torch.nn.Module,
                   x: Tensor,
                   edge_index: Tensor | SparseTensor,
                   data: Data,
                   drop_rate: float) -> typing.Union[Tensor, Tensor]:
    return model(x, edge_index, data=data, drop_rate=drop_rate)


if __name__ == "__main__":
    main()
