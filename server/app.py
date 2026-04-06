"""
FastAPI application for the Hyperlocal Inventory Curation environment.
"""

from openenv.core.env_server.http_server import create_app

try:
    from ..models import InventoryCurationAction, InventoryCurationObservation
    from .environment import HyperlocalInventoryCurationEnvironment
except ImportError:
    from models import InventoryCurationAction, InventoryCurationObservation
    from server.environment import HyperlocalInventoryCurationEnvironment


app = create_app(
    HyperlocalInventoryCurationEnvironment,
    InventoryCurationAction,
    InventoryCurationObservation,
    env_name="hyperlocal_inventory_curation_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


"""
FastAPI application for the Hyperlocal Inventory Curation environment.
"""

import os

from openenv.core.env_server.http_server import create_app

try:
    from ..models import InventoryCurationAction, InventoryCurationObservation
    from .environment import HyperlocalInventoryCurationEnvironment
except ImportError:
    from models import InventoryCurationAction, InventoryCurationObservation
    from server.environment import HyperlocalInventoryCurationEnvironment


app = create_app(
    HyperlocalInventoryCurationEnvironment,
    InventoryCurationAction,
    InventoryCurationObservation,
    env_name="hyperlocal_inventory_curation_env",
    max_concurrent_envs=4,
)


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

