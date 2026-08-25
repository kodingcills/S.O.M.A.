from __future__ import annotations

from typing import TypedDict

import numpy as np
import rerun as rr


class VesselLog(TypedDict):
    col: float
    row: float
    radius: float
    damaged: bool


_ready = False


def init() -> None:
    """Start the Rerun gRPC source and bundled web viewer once."""
    global _ready
    if _ready:
        return

    rr.init("soma")
    server_uri = rr.serve_grpc(grpc_port=9876)
    rr.serve_web_viewer(
        web_port=9090,
        open_browser=False,
        connect_to=server_uri,
    )
    _ready = True


def log_step(
    sim_id: str,
    state_vec: np.ndarray,
    step: int,
    vessels: list[VesselLog],
    error_map: np.ndarray | None = None,
) -> None:
    """Log one post-step simulation state under ``soma/{sim_id}``."""
    if not _ready:
        return

    root = f"soma/{sim_id}"
    rr.set_time("step", sequence=step)
    rr.log(
        f"{root}/tissue/integrity",
        rr.DepthImage(state_vec[:256].reshape(16, 16), meter=1.0),
    )
    rr.log(
        f"{root}/tissue/bleeding",
        rr.DepthImage(state_vec[512:768].reshape(16, 16)),
    )
    rr.log(
        f"{root}/robot/ee",
        rr.Points3D(
            state_vec[768:771].reshape(1, 3),
            colors=[[0, 180, 255]],
            radii=[0.03],
            labels=[f"EE ({sim_id})"],
        ),
    )
    rr.log(
        f"{root}/target",
        rr.Points3D(
            [[float(state_vec[776]), float(state_vec[777]), 0.0]],
            colors=[[50, 220, 80]],
            radii=[0.05],
        ),
    )

    if vessels:
        vessel_positions = np.asarray(
            [
                [vessel["col"] / 15.0, vessel["row"] / 15.0, 0.0]
                for vessel in vessels
            ],
            dtype=np.float32,
        )
        vessel_colors = [
            [220, 50, 50] if vessel["damaged"] else [80, 140, 255]
            for vessel in vessels
        ]
        rr.log(
            f"{root}/vessels",
            rr.Points3D(vessel_positions, colors=vessel_colors, radii=[0.04]),
        )

    if error_map is not None:
        rr.log(f"{root}/world_model/error", rr.DepthImage(error_map))


__all__ = ["VesselLog", "init", "log_step"]
