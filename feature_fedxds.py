# depending on your shell, you might need to add `\` before `[` and `]`.
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from resnet import resnet8
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple
from torchvision import transforms
from flwr.common import Scalar
from datasets import Dataset
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar
from flwr_datasets.partitioner import DirichletPartitioner
from heatmaps import (
    heatmaps_and_labels_zennit,
    heatmaps_and_labels_crp,
    apply_heatmap_mask,
)
from flwr.common import Metrics
from resnet import resnet8
import os
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)
from flwr.common import Context
import datasets
import os

parser = argparse.ArgumentParser(
    description="Run a federated learning simulation on CIFAR-10."
)


dirs = ["./client_noise_trained", "./client_noise_label"]

# Create the directories if they don't exist
for directory in dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")


# Add arguments
parser.add_argument(
    "--num_clients",
    default=10,
    help="Number of clients in the federated learning model",
)
parser.add_argument("--alpha", default=0.1, help="Alpha parameter for configuration")
parser.add_argument("--epochs", default=10, help="Number of epochs for training")
parser.add_argument(
    "--fraction_fit", default=1.0, help="Fraction of clients used for fitting"
)
parser.add_argument(
    "--num_gpus", type=float, default=0.5, help="Fraction of clients used for fitting"
)
parser.add_argument("--num_rounds", default=10, help="Number of FL rounds")
parser.add_argument(
    "--xai_method", type=str, default="LRP", help="If to use CRP or Zennit"
)
parser.add_argument(
    "--mask_eps", type=float, default=5.0, help="Epsilon for Pixel-flipping"
)
parser.add_argument("--sparsity", type=float, default=80.0, help="Sparsity Percentag")
args = parser.parse_args()

Net = resnet8(num_classes=10)


def batch_heatmap_to_binary_mask(batch_heatmaps):
    """
    Converts a batch of sparsified heatmaps into binary masks.

    Parameters:
    batch_heatmaps (torch.Tensor): A 3D tensor of shape (batch_size, height, width)
                                   with values between 0 and 1 after sparsification.

    Returns:
    torch.Tensor: A 3D binary mask tensor of shape (batch_size, height, width) with values 0 or 1.
    """
    # Apply thresholding across the entire batch to create binary masks
    binary_masks = torch.where(
        batch_heatmaps > 0,
        torch.tensor(1.0, device=batch_heatmaps.device),
        torch.tensor(0.0, device=batch_heatmaps.device),
    )

    return binary_masks


import torch


def make_batch_mask_dp(batch_mask: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Make a batch of binary masks differentially private through randomized response.

    Args:
        batch_mask: A batch of binary tensors (0s and 1s) of shape (batch_size, height, width)
        epsilon: Privacy parameter

    Returns:
        Privatized batch of binary masks with the same shape as batch_mask
    """
    # Compute flip probability
    p = torch.exp(torch.tensor(epsilon)) / (1 + torch.exp(torch.tensor(epsilon)))

    # Generate random values on the same device as batch_mask
    rand = torch.rand_like(batch_mask)

    # Create flips where random values exceed p
    flips = rand > p

    # Apply flips to create differentially private binary mask batch
    dp_batch_mask = torch.where(flips, 1 - batch_mask, batch_mask)

    return dp_batch_mask


tensor_loader_org = None


def train(net, trainloader, optim, epochs, cid, tensor_loader, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            # x2,y2 = next(iter(tensor_loader))
            # outputs = net(x2.cuda())
            # loss += nn.functional.cross_entropy(outputs, y2.cuda())
            loss.backward()
            optim.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["img"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


partitioner = DirichletPartitioner(
    num_partitions=args.num_clients,
    partition_by="label",
    alpha=args.alpha,
    min_partition_size=10,
    self_balancing=True,
)

mnist_fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
centralized_testset = mnist_fds.load_split("test")


def apply_transforms(batch):
    """Get transformation for MNIST dataset"""
    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])]
    )
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def apply_transforms_val(batch):
    """Get transformation for MNIST dataset"""

    # transformation to convert images to tensors and apply normalization
    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])]
    )
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, tensor_loader, cid) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net  # (num_classes=10)
        self.cid = cid
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device
        self.tensor_loader = tensor_loader

    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and conver them to a list of
        NumPy arryas. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # read from config
        lr, epochs = config["lr"], config["epochs"]

        # Define the optimizer
        optim = torch.optim.SGD(
            self.model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9
        )

        # do local training
        train(
            self.model,
            self.trainloader,
            optim,
            epochs=epochs,
            cid=self.cid,
            tensor_loader=self.tensor_loader,
            device=self.device,
        )

        if config["get_hms"]:
            self.trainloader.transform = transforms.Compose(
                [
                    ToTensor(),
                    Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
                ]
            )
            if config["xai_method"] == "LRP":
                print("USING LRP")
                hms, imgs, labels = heatmaps_and_labels_crp(
                    self.model, self.trainloader
                )
            else:
                print("USING:", config["xai_method"])
                hms, imgs, labels = heatmaps_and_labels_zennit(
                    self.model, self.trainloader, config["xai_method"]
                )
            # masked_imgs, _  = partition_image_by_heatmap(imgs, hms, percentile=30)
            print(config["sparsity_level"])
            masked_imgs, hms = apply_heatmap_mask(
                imgs, hms, sparsity_level=config["sparsity_level"]
            )
            
            torch.save(masked_imgs, f"./client_noise_trained/lrp_feat_{self.cid}.pt")
            torch.save(
                torch.tensor(labels), f"./client_noise_label/lrp_labels_{self.cid}.pt"
            )
        return self.get_parameters({}), len(self.trainloader), {}


def get_evaluate_fn(centralized_testset: Dataset):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = Net  # (num_classes=10)
        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)  # send model to device
        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms_val)

        testloader = DataLoader(testset, batch_size=64)
        # call test
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": args.epochs,  # Number of local epochs done by clients
        "lr": 0.01,
        "get_hms": False,  # Learning rate to use by clients during fit()
        "xai_method": args.xai_method,
        "sparsity_level": args.sparsity,
        "mask_eps": args.mask_eps,
    }
    if server_round == args.num_rounds:
        config["get_hms"] = True
        config["epochs"] = 10
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(Net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            Net.load_state_dict(state_dict, strict=True)

        return aggregated_parameters, aggregated_metrics


strategy = SaveModelStrategy(
    fraction_fit=args.fraction_fit,  # Sample 10% of available clients for training
    fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
    evaluate_fn=get_evaluate_fn(centralized_testset),  # global evaluation function
)

from torch.utils.data import DataLoader


def get_client_fn(dataset: FederatedDataset, tensor_loader_org):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(context: Context) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        cid = context.node_config["partition-id"]
        trainset = dataset.load_partition(int(cid))
        trainloader = DataLoader(
            trainset.with_transform(apply_transforms), batch_size=64, shuffle=True
        )
        valloader = None
        tensor_loader = None
        return FlowerClient(trainloader, valloader, tensor_loader, cid).to_client()

    return client_fn


client_fn_callback = get_client_fn(mnist_fds, tensor_loader_org)
client_resources = {"num_cpus": 2, "num_gpus": args.num_gpus}
disable_progress_bar()

history = fl.simulation.start_simulation(
    client_fn=client_fn_callback,  # a callback to construct a client
    num_clients=args.num_clients,  # total number of clients in the experiment
    config=fl.server.ServerConfig(
        num_rounds=args.num_rounds
    ),  # let's run for 10 rounds
    strategy=strategy,  # the strategy that will orchestrate the whole FL pipeline
    client_resources=client_resources,
    actor_kwargs={
        "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
    },
)

# global_accuracy_centralised = history.metrics_centralized["accuracy"]
# round = [data[0] for data in global_accuracy_centralised]
# acc = [100.0 * data[1] for data in global_accuracy_centralised]
torch.save(Net.state_dict(), './ckpt.pt')

l_labels = os.listdir("./client_noise_label")
l_labels.sort()
l_feat = os.listdir("./client_noise_trained")
l_feat.sort()

train_feat = torch.cat(
    [
        torch.load(os.path.join("./client_noise_trained", i), weights_only=True)
        for i in l_feat
    ]
)
train_labels = torch.cat(
    [
        torch.load(os.path.join("./client_noise_label", i), weights_only=True)
        for i in l_labels
    ]
)
torch.save(train_feat, "./lrp_feat_clients.pt")
torch.save(train_labels, "./lrp_labels_clients.pt")
