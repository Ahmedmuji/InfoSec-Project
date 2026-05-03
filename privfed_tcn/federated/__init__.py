"""Federated learning components."""
from .client import FLClient, client_update
from .edge_aggregator import EdgeAggregator, krum_select
from .server import FederatedServer
from .scheduler import ResourceAwareScheduler, ClientResource

__all__ = [
    "FLClient",
    "client_update",
    "EdgeAggregator",
    "krum_select",
    "FederatedServer",
    "ResourceAwareScheduler",
    "ClientResource",
]
