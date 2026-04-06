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

from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Hyperlocal Inventory Curation</title>
      <style>
        :root {
          --bg: #0b1220;
          --panel: #111a2b;
          --text: #e8eefc;
          --muted: #9fb0d1;
          --accent: #42c98f;
          --accent2: #5aa9ff;
          --border: rgba(255,255,255,0.08);
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: Georgia, "Times New Roman", serif;
          background:
            radial-gradient(circle at top left, rgba(90,169,255,0.18), transparent 32%),
            radial-gradient(circle at bottom right, rgba(66,201,143,0.16), transparent 28%),
            var(--bg);
          color: var(--text);
        }
        .wrap {
          max-width: 980px;
          margin: 0 auto;
          padding: 48px 20px 72px;
        }
        .hero {
          background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
          border: 1px solid var(--border);
          border-radius: 24px;
          padding: 32px;
          box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        }
        .eyebrow {
          display: inline-block;
          padding: 6px 12px;
          border-radius: 999px;
          background: rgba(66,201,143,0.12);
          color: var(--accent);
          font-size: 14px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }
        h1 {
          margin: 16px 0 12px;
          font-size: clamp(34px, 5vw, 56px);
          line-height: 1.05;
        }
        p.lead {
          margin: 0;
          max-width: 760px;
          color: var(--muted);
          font-size: 18px;
          line-height: 1.7;
        }
        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 16px;
          margin-top: 28px;
        }
        .card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 18px;
        }
        .card h3 {
          margin: 0 0 8px;
          font-size: 18px;
        }
        .card p {
          margin: 0;
          color: var(--muted);
          line-height: 1.6;
          font-size: 15px;
        }
        .links {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          margin-top: 28px;
        }
        a.btn {
          text-decoration: none;
          color: white;
          background: linear-gradient(135deg, var(--accent2), var(--accent));
          padding: 12px 18px;
          border-radius: 12px;
          font-weight: 600;
        }
        a.subtle {
          color: var(--text);
          border: 1px solid var(--border);
          background: rgba(255,255,255,0.03);
        }
        .footer {
          margin-top: 26px;
          color: var(--muted);
          font-size: 14px;
        }
        code {
          background: rgba(255,255,255,0.06);
          padding: 2px 6px;
          border-radius: 6px;
        }
      </style>
    </head>
    <body>
      <main class="wrap">
        <section class="hero">
          <span class="eyebrow">OpenEnv Benchmark</span>
          <h1>Hyperlocal Inventory Curation</h1>
          <p class="lead">
            A live environment for evaluating how well an agent can clean messy quick-commerce
            inventory records: normalize titles, standardize units, assign categories, merge duplicates,
            correct price anomalies, and escalate ambiguity safely.
          </p>

          <div class="grid">
            <div class="card">
              <h3>What it tests</h3>
              <p>Structured multi-step catalog curation on realistic retail inventory batches.</p>
            </div>
            <div class="card">
              <h3>Task types</h3>
              <p>Easy cleanup, medium duplicate-price repair, and hard ambiguous multi-source review.</p>
            </div>
            <div class="card">
              <h3>API-first</h3>
              <p>Use the endpoints below to reset an episode, step through actions, and inspect state.</p>
            </div>
          </div>

          <div class="links">
            <a class="btn" href="/health">Health</a>
            <a class="btn subtle" href="/docs">API Docs</a>
          </div>

          <div class="footer">
            Endpoints: <code>/health</code> <code>/reset</code> <code>/step</code> <code>/state</code>
          </div>
        </section>
      </main>
    </body>
    </html>
    """
if __name__ == "__main__":
    main()

