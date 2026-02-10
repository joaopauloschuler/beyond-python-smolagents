"""bpsa — transparent namespace alias for smolagents.

``import bpsa`` is identical to ``import smolagents``.
Sub-module access (``from bpsa.agents import CodeAgent``) works via a
meta-path finder that redirects ``bpsa.*`` imports to ``smolagents.*``.
"""

import importlib
import importlib.abc
import importlib.util
import sys

from smolagents import *  # noqa: F403
from smolagents import __version__  # noqa: F401


class _BpsaLoader(importlib.abc.Loader):
    """Loader that returns an already-imported smolagents module."""

    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        pass  # Already fully loaded


class _BpsaFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that redirects bpsa.* imports to smolagents.*."""

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("bpsa."):
            return None
        real_name = "smolagents" + fullname[len("bpsa"):]
        try:
            real_mod = importlib.import_module(real_name)
        except ImportError:
            return None
        sys.modules[fullname] = real_mod
        return importlib.util.spec_from_loader(fullname, loader=_BpsaLoader(real_mod))


sys.meta_path.insert(0, _BpsaFinder())
