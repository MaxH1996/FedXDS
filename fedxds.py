import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
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

from flwr.server.client_proxy import ClientProxy
import flwr as fl
from itertools import cycle
from collections import OrderedDict
from typing import Dict, List, Tuple
from torchvision import transforms
from flwr.common import NDArrays, Scalar

from augment import RandAugmentMC
from torchvision.transforms import v2

from resnet import resnet8
from torchvision.transforms import transforms, Normalize
from flwr_datasets.partitioner import PathologicalPartitioner, DirichletPartitioner
from flwr.common import Context
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)
import argparse
from flwr.common import Metrics
from datasets import Dataset
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar
from itertools import cycle
import datasets


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Federated Learning Experiment")
    parser.add_argument(
        "--dirichlet",
        type=float,
        default=0.1,
        help="Dirichlet alpha value for partitioning",
    )
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument(
        "--participation_rate",
        type=float,
        default=0.5,
        help="Client participation rate",
    )
    parser.add_argument(
        "--rounds", type=int, default=200, help="Number of training rounds"
    )
    parser.add_argument("--max_norm", type=int, default=10, help="Clipping Norm for DP")
    parser.add_argument("--noise", type=float, default=0.15, help="Noise for DP")
    parser.add_argument(
        "--num_gpus", type=float, default=0.2, help="Ray split over GPUs"
    )
    parser.add_argument(
        "--xai_method", type=str, default="LRP", help="Which XAI method is used"
    )
    parser.add_argument(
        "--frac_aux",
        type=float,
        default=1.0,
        help="How much of the Aux Data each Client gets.",
    )
    return parser.parse_args()


args = parse_args()


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (torch.Tensor): Tensor of images of shape (N, C, H, W).
            labels (torch.Tensor): Tensor of labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Convert image to PIL for compatibility with torchvision transforms
            # image = transforms.ToPILImage()(image)
            image = self.transform(image)
            # image = self.transform(image=image)['image']

        return image, label


# Partition dataset using command-line arguments
NUM_CLIENTS = args.num_clients
DIRICHLET_ALPHA = args.dirichlet
PARTICIPATION_RATE = args.participation_rate
MAX_NORM = args.max_norm
NOISE = args.noise
ROUNDS = args.rounds
NUM_GPUS = args.num_gpus
METHOD = args.xai_method
AUX_FRAC = args.frac_aux

print(
    "Method:",
    METHOD,
    "Num Clients:",
    NUM_CLIENTS,
    "Dir:",
    DIRICHLET_ALPHA,
    "Participation:",
    PARTICIPATION_RATE,
    "Num GPUS:",
    NUM_GPUS,
)
# Create partitioner with Dirichlet alpha
partitioner = DirichletPartitioner(
    num_partitions=NUM_CLIENTS,
    partition_by="label",
    alpha=DIRICHLET_ALPHA,
    min_partition_size=2,
    self_balancing=True,
)

Net = resnet8(num_classes=10)
#Net.load_state_dict(torch.load("./ckpt.pt", weights_only=True))

partitioner = DirichletPartitioner(
    num_partitions=NUM_CLIENTS,
    partition_by="label",
    alpha=DIRICHLET_ALPHA,
    min_partition_size=2,
    self_balancing=True,
)

mnist_fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
centralized_testset = mnist_fds.load_split("test")

v2_transform = v2.Compose(
    [
        v2.RandomAutocontrast(0.5),
        v2.RandomSolarize(0.3, 0.5),
        v2.RandomHorizontalFlip(p=0.5),
    ]
)

aux_data = torch.load("./lrp_feat_clients.pt", weights_only=True)
norm = aux_data.view(aux_data.shape[0], -1).norm(dim=1, keepdim=True)

norm = norm.view(aux_data.shape[0], 1, 1, 1)
aux_data = torch.where(norm > MAX_NORM, aux_data * (MAX_NORM / norm), aux_data)

noise = torch.empty(aux_data.shape).normal_(
    0, NOISE
)  # Generate noise for the entire tensor
aux_data += noise

aux_labels = torch.load("./lrp_labels_clients.pt", weights_only=True)

# idx = torch.randperm(50000)[:int(50000*AUX_FRAC)]

tensor_loader_org = TensorDataset(aux_data, aux_labels, transform=None)


def train(net, trainloader, optim, epochs, cid, tensor_loader, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    tensor_loader = cycle(tensor_loader)
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            x2, y2 = next(tensor_loader)
            outputs = net(x2.cuda())
            loss += nn.functional.cross_entropy(outputs, y2.cuda())
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


def apply_transforms(batch):

    transforms = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            ToTensor(),
            Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        ]
    )

    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def apply_transforms_val(batch):

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
        self.model = Net
        self.cid = cid

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tensor_loader = tensor_loader

    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})

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

        optim = torch.optim.SGD(
            self.model.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9
        )

        train(
            self.model,
            self.trainloader,
            optim,
            epochs=epochs,
            cid=self.cid,
            tensor_loader=self.tensor_loader,
            device=self.device,
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
        "epochs": 5,  # Number of local epochs done by clients
        "lr": 0.01,  # Learning rate to use by clients during fit()
    }
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

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            params_dict = zip(Net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            Net.load_state_dict(state_dict, strict=True)

        return aggregated_parameters, aggregated_metrics


strategy = SaveModelStrategy(
    fraction_fit=PARTICIPATION_RATE,
    fraction_evaluate=0.05,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=get_evaluate_fn(centralized_testset),
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
        # Let's get the partition corresponding to the i-th client
        #client_dataset = dataset.load_partition(int(cid))  # "train"
        trainset = dataset.load_partition(int(cid))

        trainloader = DataLoader(
            trainset.with_transform(apply_transforms), batch_size=64, shuffle=True
        )

        valloader = None

        tensor_loader = DataLoader(tensor_loader_org, batch_size=64, shuffle=True)

        return FlowerClient(trainloader, valloader, tensor_loader, cid).to_client()

    return client_fn


client_fn_callback = get_client_fn(mnist_fds, tensor_loader_org)

client_resources = {"num_cpus": 4, "num_gpus": NUM_GPUS}

disable_progress_bar()

history = fl.simulation.start_simulation(
    client_fn=client_fn_callback,  # a callback to construct a client
    num_clients=NUM_CLIENTS,  # total number of clients in the experiment
    config=fl.server.ServerConfig(num_rounds=ROUNDS),  # let's run for 10 rounds
    strategy=strategy,  # the strategy that will orchestrate the whole FL pipeline
    client_resources=client_resources,
    actor_kwargs={
        "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
    },
)

print(f"{history.metrics_centralized = }")

# global_accuracy_centralised = history.metrics_centralized["accuracy"]
# round = [data[0] for data in global_accuracy_centralised]
# acc = [100.0 * data[1] for data in global_accuracy_centralised]
# np.save(f'./training_accs/resnet8_{METHOD}_cifar10_{NUM_CLIENTS}_{DIRICHLET_ALPHA}_{AUX_FRAC}.npy', acc)
