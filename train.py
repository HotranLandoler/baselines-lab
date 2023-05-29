import torch
import torch.nn.functional as func
import torch_geometric.transforms as transforms
from torch_geometric.data import Data

import data_processing
import options
import utils
from models import GCN, DropGCN, MlpDropGCN
# from models import TGAT
from logger import Logger
from evaluator import Evaluator
from early_stopping import EarlyStopping

# Experiment setup
args = options.prepare_args()

# Prepare data and model
data, model = utils.prepare_data_and_model(args)

print(model)
weight = torch.tensor([1, args.loss_weight], dtype=torch.float32, device=args.device)
evaluator = Evaluator(args.metrics, num_classes=args.num_classes)
logger = Logger(settings=args)

print(f"Train started with setting {args}")
for run in range(args.runs):
    print(f"Run {run} " + "-" * 20)
    # gc.collect()

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        optimizer.zero_grad()

        if args.model == "dropgcn":
            output = model(data.x, data.adj_t, drop_rate=args.drop_rate)
        else:
            output = model(data.x, data.adj_t)

        loss = func.nll_loss(output[data.train_mask], data.y[data.train_mask],
                             weight=weight)
        loss.backward()
        optimizer.step()
        # print(f"Epoch {epoch} finished. loss: {loss.item():>7f}")

        # Validation
        model.eval()
        predicts = model(data.x, data.adj_t)
        val_loss = func.nll_loss(predicts[data.valid_mask], data.y[data.valid_mask],
                                 weight=weight)
        print(f"Epoch {epoch} finished. train_loss: {loss.item():>7f} val_loss: {val_loss.item():>7f}")
        if (args.use_early_stopping and
                early_stopping.check_stop(val_loss.item())):
            print("Early Stopping")
            break

    # Test
    model.eval()
    predicts = model(data.x, data.adj_t)
    # result = metric(predicts[dataset.test_mask], dataset.y[dataset.test_mask], num_classes=OUT_FEATS)
    # logger.add_result(result.item())
    print(f"Run {run}: ", end='')
    for result in evaluator.evaluate_test(predicts, data.y, data.test_mask):
        print(result, end='')
    print("")

print("Train ended.")
for result in evaluator.get_final_results():
    print(result, end='')
print("")
if args.save_log:
    logger.save_to_file(evaluator, path=args.log_path)
