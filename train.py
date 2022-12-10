import gc

import torch
import torch.nn.functional as func
from cogdl.data import Graph

import data_processing
import models
import options
from logger import Logger
from evaluator import Evaluator


# Prepare data
dataset: Graph = data_processing.DGraphDataset()[0]
data_processing.data_preprocess(dataset)

# Experiment setup
# OUT_FEATS = 2
# HIDDEN_SIZE = 64
# NUM_LAYERS = 2
# DROPOUT = 0.0
args = options.prepare_args()

model = models.GCN(in_feats=dataset.num_features, hidden_size=args.hidden_size,
                   out_feats=args.num_classes, dropout=args.dropout, num_layers=args.num_layers)
weight = torch.tensor([1, args.loss_weight]).float()
evaluator = Evaluator(args.metrics, num_classes=args.num_classes)
logger = Logger(settings=args)

print(f"Train started with setting {args}")
for run in range(args.runs):
    print(f"Run {run} " + "-" * 20)
    gc.collect()

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(dataset)
        loss = func.nll_loss(output[dataset.train_mask], dataset.y[dataset.train_mask],
                             weight=weight)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} finished. loss: {loss.item():>7f}")
        # model.eval()
        # predicts = model(dataset)
        # result = metric(predicts[dataset.val_mask], dataset.y[dataset.val_mask])
        # print(f"Epoch {epoch} finished. loss: loss.item():>7f AUC_valid: {result:.6f}")

    # Test
    model.eval()
    predicts = model(dataset)
    # result = metric(predicts[dataset.test_mask], dataset.y[dataset.test_mask], num_classes=OUT_FEATS)
    # logger.add_result(result.item())
    print(f"Run {run}: ", end='')
    for result in evaluator.evaluate_test(predicts, dataset.y, dataset.test_mask):
        print(result, end='')
    print("")

print("Train ended.")
for result in evaluator.get_final_results():
    print(result, end='')
print("")
if args.save_log:
    logger.save_to_file(evaluator, path=args.log_path)
