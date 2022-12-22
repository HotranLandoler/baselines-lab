import gc

import torch
import torch.nn.functional as func
from torch import Tensor
# from cogdl.data import Graph

import data_processing
import options
from data_processing import DynamicGraph
from models import GCN, TGAT
from logger import Logger
from evaluator import Evaluator

# Experiment setup
args = options.prepare_args()
device = torch.device(args.device)

# Prepare data
is_tgat = args.model == "tgat"
dataset: DynamicGraph = data_processing.DGraphDataset(to_undirected=not is_tgat)[0]

if is_tgat:
    dataset = data_processing.process_tgat_data(dataset)
    model = TGAT(in_channels=17, out_channels=2)
else:
    data_processing.data_preprocess(dataset)
    model = GCN(in_feats=dataset.num_features, hidden_size=args.hidden_size,
                out_feats=args.num_classes, dropout=args.dropout, num_layers=args.num_layers)

dataset = dataset.to(device)
model.to(device)
weight = torch.tensor([1, args.loss_weight]).to(device).float()
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
