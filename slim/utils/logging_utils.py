import logging

import ray
import wandb

from . import wandb_utils

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def build_sglang_metrics_endpoints(args, servers):
    """Build the W&B Prometheus scrape endpoints for SGLang.

    Always includes each model's router metrics (the router exposes
    ``/metrics`` on ``prometheus_port`` regardless of engine config). When
    ``--sglang-enable-metrics`` is set, each engine's own ``/metrics`` endpoint
    (which carries scheduler throughput such as ``sglang:gen_throughput``) is
    added too.

    Returns a name -> url dict suitable for ``x_stats_open_metrics_endpoints``,
    or ``None`` when there is nothing to scrape.
    """
    endpoints: dict[str, str] = {}
    enable_engine_metrics = getattr(args, "sglang_enable_metrics", False)

    for name, server in (servers or {}).items():
        prom_port = getattr(server, "router_prometheus_port", None)
        if server.router_ip is not None and prom_port is not None:
            endpoints[f"sgl_router_{name}"] = f"http://{server.router_ip}:{prom_port}/metrics"

        if enable_engine_metrics:
            urls = ray.get([e.get_url.remote() for e in server.engines if e is not None])
            for idx, url in enumerate(u for u in urls if u is not None):
                endpoints[f"sgl_engine_{name}_{idx}"] = f"{url}/metrics"

    return endpoints or None


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


def log(args, metrics):
    if args.use_wandb:
        wandb.log(metrics)
