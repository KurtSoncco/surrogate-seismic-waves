from __future__ import annotations

import json
from pathlib import Path

from scoring.score_ship_gates import score_one


def test_score_ship_gates_beats_and_misses(tmp_path: Path):
    win = {
        "name": "winner",
        "test_by_domain": {
            "iid": {"rel_l2_TF": 0.30},
            "ood_dipping": {"rel_l2_TF": 0.32},
            "ood_three_layer": {"rel_l2_TF": 0.50},
        },
        "three_layer_kill": False,
    }
    lose = {
        "name": "lose",
        "test_by_domain": {
            "iid": {"rel_l2_TF": 0.30},
            "ood_dipping": {"rel_l2_TF": 0.32},
            "ood_three_layer": {"rel_l2_TF": 0.54},
        },
        "three_layer_kill": True,
    }
    p_win = tmp_path / "win.json"
    p_lose = tmp_path / "lose.json"
    p_win.write_text(json.dumps(win))
    p_lose.write_text(json.dumps(lose))
    s_win = score_one(p_win)
    s_lose = score_one(p_lose)
    assert s_win["ship"] is True
    assert s_win["beats_ship_3l"] is True
    assert s_lose["ship"] is False
    assert s_lose["three_layer_kill"] is True
