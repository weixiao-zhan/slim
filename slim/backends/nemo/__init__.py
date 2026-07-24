# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo AutoModel training backend."""

from importlib import import_module

__all__ = [
    "ActorNeMoTrainer",
    "CriticNeMoTrainer",
    "NeMoTrainer",
    "nemo_parse_args",
    "validate_args",
]

_EXPORTS = {
    "ActorNeMoTrainer": (".actor", "ActorNeMoTrainer"),
    "CriticNeMoTrainer": (".critic", "CriticNeMoTrainer"),
    "NeMoTrainer": (".base", "NeMoTrainer"),
    "nemo_parse_args": (".arguments", "nemo_parse_args"),
    "validate_args": (".arguments", "validate_args"),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
