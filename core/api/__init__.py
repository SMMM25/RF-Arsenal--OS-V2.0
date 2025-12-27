"""
RF Arsenal OS - REST/WebSocket API Module
Provides API endpoints for remote control and monitoring.
Offline-safe with local-only binding options.
"""

from .rest_api import RestAPI, APIEndpoint
from .websocket_server import WebSocketServer, WSClient
from .api_security import APIAuth, TokenManager

__all__ = [
    'RestAPI',
    'APIEndpoint',
    'WebSocketServer',
    'WSClient',
    'APIAuth',
    'TokenManager'
]

__version__ = "1.0.0"
