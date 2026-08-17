from __future__ import annotations

from unittest.mock import MagicMock, patch

import wandb_util


def test_run_tags_order():
    tags = wandb_util._run_tags(
        {"mix": "M2100", "encoder": "gno", "fno_kind": "vanilla", "host": "lambda"}
    )
    assert tags == ["M2100", "gno", "vanilla", "lambda"]


def test_run_tags_skips_empty():
    assert wandb_util._run_tags({"mix": "M700", "encoder": "", "host": None}) == ["M700"]


def test_init_wandb_tags_group_and_epoch_metrics():
    fake = MagicMock()
    fake.init.return_value = object()
    cfg = {
        "mix": "M2100",
        "encoder": "gno",
        "fno_kind": "vanilla",
        "host": "lambda",
        "lr": 1e-3,
    }
    with patch.dict("os.environ", {"WANDB_PROJECT": "deeponet-nscale", "WANDB_API_KEY": "k"}, clear=False):
        with patch.dict("sys.modules", {"wandb": fake}):
            run = wandb_util.init_wandb("M2100_gino_wide_lambda", cfg)
    assert run is fake.init.return_value
    kwargs = fake.init.call_args.kwargs
    assert kwargs["project"] == "deeponet-nscale"
    assert kwargs["name"] == "M2100_gino_wide_lambda"
    assert kwargs["tags"] == ["M2100", "gno", "vanilla", "lambda"]
    assert kwargs["group"] == "M2100"
    fake.define_metric.assert_any_call("epoch")
    fake.define_metric.assert_any_call("train/*", step_metric="epoch")
    fake.define_metric.assert_any_call("val/*", step_metric="epoch")
    fake.define_metric.assert_any_call("lr", step_metric="epoch")


def test_init_wandb_disabled():
    assert wandb_util.init_wandb("x", {}, enabled=False) is None
