"""Offline HTTP transport for tests that exercise real provider SDKs."""

import asyncio
import socket
from uuid import uuid4

import pytest


@pytest.fixture
def simulated_http(monkeypatch):
    import httpx

    for name in ("OPENAI_API_KEY", "CO_API_KEY", "COHERE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "offline-test-key")
    # Unique URLs isolate LangChain's cached HTTP clients between test cases.
    host = f"{uuid4().hex}.invalid"
    monkeypatch.setenv("OPENAI_BASE_URL", f"https://{host}/v1")
    monkeypatch.setenv("OPENAI_API_BASE", f"https://{host}/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"https://{host}")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    def forbid_network(*args, **kwargs):
        raise AssertionError("Simulation attempted a real network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    clients = []

    def install(handler):
        for client_type in (httpx.Client, httpx.AsyncClient):
            original_init = client_type.__init__

            def init(self, *args, _original_init=original_init, **kwargs):
                kwargs.update(transport=httpx.MockTransport(handler), trust_env=False)
                _original_init(self, *args, **kwargs)
                clients.append(self)

            monkeypatch.setattr(client_type, "__init__", init)

    yield install
    for client in clients:
        if isinstance(client, httpx.AsyncClient):
            asyncio.run(client.aclose())
        else:
            client.close()
