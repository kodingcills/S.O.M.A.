from __future__ import annotations

import asyncio
from typing import Final

import numpy as np

from backend.event_log import write_event

NOOP: Final = 0
LEFT: Final = 1
RIGHT: Final = 2
UP: Final = 3
DOWN: Final = 4
DO: Final = 5
SLEEP: Final = 6
MAKE_WOOD_PICKAXE: Final = 11

MAP_ROWS: Final = 9
MAP_COLS: Final = 11
PLAYER_ROW: Final = 4
PLAYER_COL: Final = 5
LOW_STAT: Final = 0.25
WATER_BLOCKS: Final = frozenset({3, 24})
FOOD_BLOCKS: Final = frozenset({16})
TABLE_BLOCKS: Final = frozenset({11})
RESOURCE_BLOCKS: Final = frozenset({3, 4, 5, 8, 9, 10, 16, 21, 22, 23, 24})


class ReactiveAgent:
    def __init__(self, driver) -> None:
        self._driver = driver

    @staticmethod
    def _nearest(
        observation: dict[str, np.ndarray], blocks: frozenset[int]
    ) -> tuple[int, int, int] | None:
        map_tokens = observation["token_2d"].reshape(MAP_ROWS, MAP_COLS, 4)
        positions = np.argwhere(np.isin(map_tokens[:, :, 0], tuple(blocks)))
        if positions.size == 0:
            return None
        distances = np.abs(positions[:, 0] - PLAYER_ROW) + np.abs(
            positions[:, 1] - PLAYER_COL
        )
        row, col = positions[int(np.argmin(distances))]
        return int(row), int(col), int(distances.min())

    @staticmethod
    def _move(row: int, col: int, distance: int) -> int:
        if distance <= 1:
            return DO
        row_delta = row - PLAYER_ROW
        col_delta = col - PLAYER_COL
        if abs(col_delta) >= abs(row_delta):
            return RIGHT if col_delta > 0 else LEFT
        return DOWN if row_delta > 0 else UP

    def _toward(
        self, observation: dict[str, np.ndarray], blocks: frozenset[int]
    ) -> int:
        nearest = self._nearest(observation, blocks)
        if nearest is None:
            return NOOP
        return self._move(*nearest)

    def _select_action(
        self,
        observation: dict[str, np.ndarray],
        stats: dict[str, float],
    ) -> int:
        if stats["energy"] < LOW_STAT:
            return SLEEP
        if stats["drink"] < LOW_STAT:
            return self._toward(observation, WATER_BLOCKS)
        if stats["food"] < LOW_STAT:
            return self._toward(observation, FOOD_BLOCKS)
        vector = observation["vector"]
        table = self._nearest(observation, TABLE_BLOCKS)
        if float(vector[0]) >= 1.0 and table is not None:
            if table[2] <= 1:
                return MAKE_WOOD_PICKAXE
            return self._move(*table)
        return self._toward(observation, RESOURCE_BLOCKS)

    async def run_step(self) -> None:
        observation = self._driver.current_obs_tokens()
        stats = self._driver.env.game_stats()
        action = self._select_action(observation, stats)
        step = await asyncio.to_thread(self._driver.step_with_action, action)
        achievements = self._driver.env.achievements()
        write_event(
            "simulation_step",
            sim_id="comparison",
            step=int(step.step),
            action=int(step.action),
            reward=float(step.reward),
            done=bool(step.done),
            stats=dict(step.stats),
            achievements=np.asarray(achievements, dtype=bool).tolist(),
            n_achievements=int(np.asarray(achievements, dtype=bool).sum()),
        )
        if step.done:
            await asyncio.to_thread(self._driver.reset, 42)

    async def run(self) -> None:
        await asyncio.to_thread(self._driver.reset, 42)
        while True:
            await self.run_step()
            await asyncio.sleep(0.05)
