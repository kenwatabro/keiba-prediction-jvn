import sys

from _script_compat import load_role_script


_is_main = __name__ == "__main__"
_module = load_role_script("mini_raceday", "predict_today_netkeiba_weights.py")
sys.modules[__name__] = _module
globals().update(vars(_module))

if _is_main:
    _module.main()
