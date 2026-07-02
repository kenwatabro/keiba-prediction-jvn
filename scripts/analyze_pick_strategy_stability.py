import sys

from _script_compat import load_role_script


_is_main = __name__ == "__main__"
_module = load_role_script("workstation_ml", "analyze_pick_strategy_stability.py")
sys.modules[__name__] = _module
globals().update(vars(_module))

if _is_main:
    _module.main()
