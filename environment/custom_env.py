"""WasteFlow Gymnasium environment.

Implemented in Trello cards 03 (simulation core) and 04 (Gymnasium env, OBS_SPEC, action mask, reward).

Design summary (see the "Approved design decisions" card):
    - 15 collection locations, 3 vehicles, 1 disposal facility, 8-hour shift.
    - Event-driven: one idle vehicle decides per step.
    - Discrete(17) actions: 0-14 go to location i, 15 go to disposal, 16 wait.
    - Dict observation: {"observation": Box[0, 1], "action_mask": MultiBinary(17)}.
"""
