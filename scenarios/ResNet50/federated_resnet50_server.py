# federated_server

import os
import argparse

import flwr as fl


parser = argparse.ArgumentParser(
    description="Federated Learning Flower based training")
parser.add_argument(
    "-b", "--server_addr",
    type=str,
    default="localhost",
    help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
parser.add_argument(
    "-p", "--server_port",
    type=str,
    default="29500",
    help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")
parser.add_argument(
    "-i", "--interface",
    type=str,
    default="eth0",
    help="""Interface that current device is listening on. It will default to eth0 if 
    not provided.""")

args = parser.parse_args()


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    os.environ['SERVER_ADDR'] = args.server_addr
    os.environ['SERVER_PORT'] = args.server_port
    os.environ['GLOO_SOCKET_IFNAME'] = args.interface
    os.environ["TP_SOCKET_IFNAME"] = args.interface
    fl.server.start_server(
        server_address=f"{args.server_addr}:{args.server_port}",
        config=fl.server.ServerConfig(num_rounds=40),
        strategy=fl.server.strategy.FedAvg(
            evaluate_metrics_aggregation_fn=weighted_average
        )
    )