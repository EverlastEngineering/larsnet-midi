"""
Schema-driven CLI argument builder.

Builds an argparse ArgumentParser from the SETTINGS_REGISTRY in
webui.settings_schema, so a new SettingDefinition with a non-empty
``cli_flag`` automatically becomes a CLI flag — no parallel list of
argparse calls to keep in sync.

Public surface
--------------
- :func:`build_cli_parser` — returns a fully populated ``ArgumentParser``.
- :func:`apply_cli_overrides` — given parsed args and a config dict,
  walks the registry, and for each flag that was actually provided on
  the command line, writes the value into ``config[yaml_path...]``.
- :func:`count_cli_flags` — number of registry entries with a non-empty
  ``cli_flag``, used for the startup banner.

Design notes
------------
- CLI defaults are the schema's ``default`` so that ``argparse --help``
  shows the same number as the schema/webui form.
- ``--learn``, ``--maxtime``, ``--stems``, ``--project`` are
  orchestration flags — they live alongside the schema-driven ones
  but are added by the caller (see ``stems_to_midi_cli.py``).
- Boolean settings become ``--foo / --no-foo`` pairs so users can
  explicitly turn them off (mirroring how the WebUI checkbox works).
- Validation runs through ``SettingDefinition.validate()`` after parse.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

from .settings_schema import (
    SETTINGS_REGISTRY,
    SettingDefinition,
    SettingType,
)


# Map schema type → callable that converts a CLI string to the right Python type.
# argparse handles int/float natively; bool needs custom handling via --foo/--no-foo.
_TYPE_FACTORIES = {
    SettingType.BOOL: argparse.BooleanOptionalAction,
}


def _python_type(definition: SettingDefinition):
    """Return the callable argparse should use to coerce the value."""
    if definition.type == SettingType.BOOL:
        return None  # handled via BooleanOptionalAction
    if definition.type == SettingType.INT:
        return int
    if definition.type == SettingType.FLOAT:
        return float
    # STRING, PATH, CHOICE — leave as str
    return str


def _format_help(definition: SettingDefinition) -> str:
    """Build the help text for a CLI flag from the schema definition."""
    base = definition.description or definition.label
    bits: List[str] = [base]
    if definition.unit:
        bits.append(f"(unit: {definition.unit})")
    if definition.allowed_values:
        bits.append(
            "choices: " + ", ".join(str(v) for v in definition.allowed_values)
        )
    if definition.min_value is not None or definition.max_value is not None:
        lo = definition.min_value
        hi = definition.max_value
        bits.append(f"range: [{lo}, {hi}]")
    if definition.default is not None:
        bits.append(f"default: {definition.default}")
    return " ".join(bits)


def _add_one_flag(
    parser: argparse.ArgumentParser,
    definition: SettingDefinition,
    dest_override: Optional[str] = None,
) -> str:
    """Add a single CLI flag to the parser based on its schema definition.

    Returns the dest name argparse will use on the Namespace.
    """
    flag = definition.cli_flag
    if not flag:
        return ""  # schema marks this as not-a-CLI-flag

    # The dest must match the schema key, otherwise apply_cli_overrides can't
    # find the value on the parsed Namespace. argparse normally derives dest
    # from the flag (e.g. --kick-geomean → kick_geomean), which doesn't match
    # the schema key 'kick_geomean_threshold'. We force dest=key.
    dest = dest_override or definition.key

    if definition.type == SettingType.BOOL:
        # --foo / --no-foo lets the user explicitly turn a bool on or off.
        parser.add_argument(
            flag,
            action=argparse.BooleanOptionalAction,
            default=definition.default,
            dest=dest,
            help=_format_help(definition),
        )
        return dest

    kwargs: Dict[str, Any] = {
        "type": _python_type(definition),
        "default": definition.default,
        "dest": dest,
        "help": _format_help(definition),
    }
    if definition.allowed_values:
        kwargs["choices"] = list(definition.allowed_values)

    # Nullable numeric settings: empty string should map to None
    if definition.nullable and definition.type in (
        SettingType.INT,
        SettingType.FLOAT,
        SettingType.STRING,
        SettingType.PATH,
        SettingType.CHOICE,
    ):
        original_type = kwargs["type"]
        original_default = kwargs["default"]

        def _coerce(value, _t=original_type, _d=original_default):  # noqa: ANN001
            if isinstance(value, str) and value.strip() == "":
                return None
            if value is None:
                return _d
            return _t(value)

        kwargs["type"] = _coerce

    parser.add_argument(flag, **kwargs)
    return dest


def build_cli_parser(
    *,
    prog: Optional[str] = None,
    description: Optional[str] = None,
    extra_args: Optional[List[argparse.ArgumentParser.add_argument_group]] = None,
) -> argparse.ArgumentParser:
    """
    Build an ``argparse.ArgumentParser`` from every registry entry that
    has a non-empty ``cli_flag``.

    Args:
        prog: Program name (default: sys.argv[0]).
        description: Optional description line.
        extra_args: Optional list of (group_title, [(flag, kwargs), ...]) tuples
            to add orchestration flags. Unused for now — kept for future
            extension.

    Returns:
        Populated ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description
        or "Drum-stem-to-MIDI conversion. Every setting that exposes "
        "a CLI flag is generated from the centralized settings schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for definition in SETTINGS_REGISTRY:
        if definition.cli_flag:
            _add_one_flag(parser, definition)
    return parser


def count_cli_flags() -> int:
    """Return the number of registry entries exposed as CLI flags."""
    return sum(1 for s in SETTINGS_REGISTRY if s.cli_flag)


def apply_cli_overrides(
    args: argparse.Namespace,
    config: Dict[str, Any],
    *,
    only_set: bool = True,
) -> Tuple[int, List[str]]:
    """
    Walk the parsed args; for every schema entry whose flag was actually
    provided on the command line, write the value into the config dict at
    ``definition.yaml_path``.

    Args:
        args: Result of ``parser.parse_args()``.
        config: In-place config dict (mutated).
        only_set: If True, only override keys whose CLI value differs from
            the schema default. If False, write every flag (default behavior
            is fine because argparse fills missing flags with the schema
            default anyway).

    Returns:
        Tuple of (overrides_applied, list_of_keys_applied).
    """
    applied: List[str] = []
    for definition in SETTINGS_REGISTRY:
        if not definition.cli_flag or not definition.yaml_path:
            continue
        # We set dest=definition.key in the parser, so the attribute is
        # always available on the Namespace.
        if not hasattr(args, definition.key):
            continue
        value = getattr(args, definition.key)

        if only_set and value == definition.default:
            # argparse filled this in with the schema default → user didn't
            # actually pass the flag. Skip so we don't pollute the config.
            continue

        # Walk the yaml_path, creating dicts as needed.
        cursor = config
        path = list(definition.yaml_path)
        for key in path[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = value
        applied.append(definition.key)

    return len(applied), applied


def validate_args(
    args: argparse.Namespace,
) -> List[str]:
    """
    Run ``SettingDefinition.validate`` on every value the user supplied.

    Returns a list of human-readable error messages. Empty list = valid.
    """
    errors: List[str] = []
    for definition in SETTINGS_REGISTRY:
        if not definition.cli_flag:
            continue
        if not hasattr(args, definition.key):
            continue
        value = getattr(args, definition.key)
        if value is None and definition.nullable:
            continue
        ok, err = definition.validate(value)
        if not ok:
            errors.append(f"{definition.cli_flag}: {err}")
    return errors
