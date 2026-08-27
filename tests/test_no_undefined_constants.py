"""Every CONFIG_CONSTANT a module uses must actually be imported into it.

Shipped to customers in v1.6 and crashed every run about 15 seconds in:

    File "main.py", line 927, in run
    NameError: name 'SAFE_ZONE_ENEMY_QUIET_SECONDS' is not defined

The constant was added to config.py and used in main.py's frame loop, but the
line adding it to main's `from config import (...)` never landed. Nothing
caught it:

* `import main` succeeds -- the name is only looked up when that branch of
  run() executes, and importing never runs it.
* The whole test suite stayed green, because no test drives run()'s loop.
  It cannot: the loop needs a live game window.

So the module imported fine, the suite passed, the build succeeded, and the
bot died a few seconds into every customer's session.

This test closes that hole for the whole class of bug. It parses each module,
finds every ALL_CAPS name the code READS, and checks the module can actually
resolve it. That is the config-constant convention this project uses
everywhere, and a missing import of one is invisible until the exact line
runs -- which, in a frame loop guarded by state, can be minutes in or never
during testing.
"""

import ast
import importlib
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_CONST = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Modules that pull constants from config. gui is included deliberately --
# importing it only defines classes, it does not build a window.
MODULES = ["main", "combat", "detectors", "gui", "input_control", "minimap",
           "skills", "targeting", "fusion", "hud_ocr", "satanic_name"]


def _constants_read(source):
    """ALL_CAPS names the source READS, minus the ones it provides itself.

    "Provides" covers more than plain assignment: a ctypes `class INPUT`, and
    an import done INSIDE a function (main.py does this for the enemy tiers)
    both bind the name perfectly well and must not be reported.
    """
    tree = ast.parse(source)
    read, bound = set(), set()

    def _bind(name):
        if name and _CONST.match(name):
            bound.add(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if not _CONST.match(node.id):
                continue
            if isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            else:
                read.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                _bind(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            _bind(node.name)
        elif isinstance(node, ast.Global):
            for name in node.names:
                _bind(name)
    return read - bound


@pytest.mark.parametrize("module_name", MODULES)
def test_every_constant_a_module_uses_resolves(module_name):
    path = os.path.join(_ROOT, module_name + ".py")
    if not os.path.exists(path):
        pytest.skip(f"{module_name}.py not present")
    with open(path, encoding="utf-8") as fh:
        used = _constants_read(fh.read())

    module = importlib.import_module(module_name)
    missing = sorted(
        name for name in used
        if not hasattr(module, name) and not hasattr(__builtins__, name)
        and name not in dir(__builtins__)
    )
    assert missing == [], (
        f"{module_name}.py reads these constants but cannot resolve them: "
        f"{missing}. Add them to its `from config import (...)`. This is the "
        f"v1.6 NameError crash -- it will not surface until that exact line "
        f"runs on a customer's machine.")


def test_the_checker_actually_catches_a_missing_import():
    """A guard that never fires is worse than none -- prove it fires."""
    source = (
        "from config import KNOWN_ONE\n"
        "def f():\n"
        "    if x < MISSING_CONSTANT:\n"
        "        return KNOWN_ONE\n"
    )
    # KNOWN_ONE is imported, so it is fine. MISSING_CONSTANT is the v1.6 bug
    # in miniature: read inside a function body, bound nowhere.
    assert _constants_read(source) == {"MISSING_CONSTANT"}


def test_the_checker_accepts_an_import_inside_a_function():
    """main.py imports the enemy tiers inside run(). That binds them."""
    source = (
        "def f():\n"
        "    from detectors import ENEMY_TIER_RED\n"
        "    return ENEMY_TIER_RED\n"
    )
    assert _constants_read(source) == set()


def test_the_checker_ignores_constants_the_module_defines_itself():
    source = (
        "LOCALLY_DEFINED = 3\n"
        "def f():\n"
        "    return LOCALLY_DEFINED\n"
    )
    assert _constants_read(source) == set()
