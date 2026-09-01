"""game2048 exception hierarchy."""

from __future__ import annotations


class Game2048Error(Exception):
    """Base class for every game2048 error."""


class InvalidActionError(Game2048Error):
    """A decision outside the four slides {0=Up, 1=Down, 2=Right, 3=Left}.

    Note this is NOT raised for a slide that happens to change nothing — that
    is a legal no-op at the mdp layer, and whether the agent may choose it is
    the gym's ``action_mode``.
    """
