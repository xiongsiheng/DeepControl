from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


def merge_utility_curve(
    novelty_curve: List[float],
    effectiveness_curve: List[float],
    gamma: float,
) -> List[float]:
    L = min(len(novelty_curve), len(effectiveness_curve))
    if L == 0:
        return []
    return [gamma * novelty_curve[i] + (1.0 - gamma) * effectiveness_curve[i] for i in range(L)]


@dataclass
class OfflineUtilityConfig:
    gamma: float = 0.5
    stop_threshold: Optional[float] = None
    stop_patience: int = 2


@dataclass
class OfflineUtilityResult:
    novelty_curve: List[float]
    effectiveness_curve: List[float]
    utility_curve: List[float]
    suggested_stop_turn: Optional[int]


class OfflineUtilityScorer:
    def __init__(self, config: OfflineUtilityConfig):
        self.config = config

    def _suggest_stop_turn(self, utility_curve: List[float]) -> Optional[int]:
        if not utility_curve:
            return None
        if self.config.stop_threshold is None:
            return int(max(range(len(utility_curve)), key=lambda i: utility_curve[i])) + 1

        low = 0
        for i, u in enumerate(utility_curve):
            if u < self.config.stop_threshold:
                low += 1
                if low >= self.config.stop_patience:
                    return i + 1
            else:
                low = 0
        return None

    def score(
        self,
        novelty_curve: List[float],
        effectiveness_curve: List[float],
    ) -> OfflineUtilityResult:
        utility_curve = merge_utility_curve(
            novelty_curve=novelty_curve,
            effectiveness_curve=effectiveness_curve,
            gamma=self.config.gamma,
        )
        return OfflineUtilityResult(
            novelty_curve=novelty_curve,
            effectiveness_curve=effectiveness_curve,
            utility_curve=utility_curve,
            suggested_stop_turn=self._suggest_stop_turn(utility_curve),
        )


@dataclass
class OnlineUtilityConfig:
    gamma: float = 0.5
    low_utility_threshold: float = 0.10
    min_turns_before_stop: int = 2
    low_utility_patience: int = 2
    force_stop_after_turn: Optional[int] = None
    continue_threshold: float = 0.25
    continue_patience: int = 2
    continue_score_max: float = -2.0
    continue_prob: float = 0.10
    random_seed: Optional[int] = None


@dataclass
class OnlineUtilityState:
    turn_idx: int = 0
    low_utility_streak: int = 0
    high_utility_streak: int = 0
    intervention_sent: bool = False


@dataclass
class OnlineUtilityDecision:
    utility: float
    decision: str
    reason: str
    control_text: Optional[str]
    tools_allowed: List[str]


class OnlineUtilityController:
    """
    Real-time stop/continue decision:
    - soft guidance text for model
    - hard tool gating for runtime
    """

    def __init__(self, config: OnlineUtilityConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)

    def compute_utility(self, novelty: float, effectiveness: float) -> float:
        return float(self.config.gamma * novelty + (1.0 - self.config.gamma) * effectiveness)

    def decide(
        self,
        state: OnlineUtilityState,
        novelty: float,
        effectiveness: float,
        score_t: Optional[float] = None,
        current_tools: Optional[List[str]] = None,
        intervention_enabled: bool = True,
    ) -> OnlineUtilityDecision:
        state.turn_idx += 1
        utility = self.compute_utility(novelty, effectiveness)

        if utility < self.config.low_utility_threshold:
            state.low_utility_streak += 1
            state.high_utility_streak = 0
        else:
            state.low_utility_streak = 0
            if utility >= self.config.continue_threshold:
                state.high_utility_streak += 1
            else:
                state.high_utility_streak = 0

        force_stop = (
            self.config.force_stop_after_turn is not None
            and state.turn_idx >= self.config.force_stop_after_turn
        )
        patience_stop = (
            state.turn_idx >= self.config.min_turns_before_stop
            and state.low_utility_streak >= self.config.low_utility_patience
        )
        stop = force_stop or patience_stop

        if current_tools is None:
            current_tools = ["search", "expand"]

        if state.intervention_sent:
            return OnlineUtilityDecision(
                utility=utility,
                decision="monitor_only",
                reason="one_shot_intervention_already_sent",
                control_text=None,
                tools_allowed=list(current_tools),
            )

        if not intervention_enabled:
            return OnlineUtilityDecision(
                utility=utility,
                decision="no_intervention",
                reason="intervention_gate_off",
                control_text=None,
                tools_allowed=list(current_tools),
            )

        if stop:
            tools_allowed = [t for t in current_tools if t != "search"]
            if not tools_allowed:
                tools_allowed = []
            reason = "utility_below_threshold" if patience_stop else "reach_max_search_turn"
            control_text = "<control>Stop searching</control>"
            state.intervention_sent = True
            return OnlineUtilityDecision(
                utility=utility,
                decision="stop",
                reason=reason,
                control_text=control_text,
                tools_allowed=tools_allowed,
            )

        should_continue_intervene = (
            state.high_utility_streak >= self.config.continue_patience
            and score_t is not None
            and score_t < self.config.continue_score_max
            and self.rng.random() < self.config.continue_prob
        )
        if should_continue_intervene:
            state.intervention_sent = True
            return OnlineUtilityDecision(
                utility=utility,
                decision="continue_once",
                reason="sampled_continue_intervention",
                control_text="<control>Continue the search for one additional step</control>",
                tools_allowed=list(current_tools),
            )

        return OnlineUtilityDecision(
            utility=utility,
            decision="no_intervention",
            reason="utility_monitored",
            control_text=None,
            tools_allowed=list(current_tools),
        )
