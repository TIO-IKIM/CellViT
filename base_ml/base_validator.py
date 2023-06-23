# -*- coding: utf-8 -*-
# Validators
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from schema import Schema, Or

sweep_schema = Schema(
    {
        "method": Or("grid", "random", "bayes"),
        "name": str,
        "metric": {"name": str, "goal": Or("maximize", "minimize")},
        "run_cap": int,
    },
    ignore_extra_keys=True,
)
