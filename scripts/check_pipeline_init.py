"""Utility to require DEFAULT_PIPELINE.args to be set before running scripts.

Usage:
    from scripts.check_pipeline_init import require_pipeline_args
    require_pipeline_args()

This raises RuntimeError with guidance if not set.
"""
from src.pipeline import DEFAULT_PIPELINE


def require_pipeline_args():
    if DEFAULT_PIPELINE is None:
        raise RuntimeError("DEFAULT_PIPELINE is not initialized. Ensure src.pipeline.DEFAULT_PIPELINE is available.")

    if DEFAULT_PIPELINE.args is None:
        # try to auto-inject args by calling a common get_args if available
        try:
            # try to import caller's script-level get_args if present
            import importlib
            caller = importlib.import_module('scripts.train_model')
            if hasattr(caller, 'get_args'):
                DEFAULT_PIPELINE.set_args(caller.get_args())
        except Exception:
            # nothing we can do automatically; raise helpful guidance
            raise RuntimeError(
                "Pipeline args are not set. Call DEFAULT_PIPELINE.set_args(args) at script entry before invoking pipeline-dependent functions, "
                "or expose a get_args(argv=None) in your script so the check utility can auto-inject args for tests."
            )
    return True
