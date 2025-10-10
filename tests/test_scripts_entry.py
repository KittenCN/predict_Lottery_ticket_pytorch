import importlib
import sys
from src.pipeline import DEFAULT_PIPELINE


def test_predict_main_sets_pipeline_args(monkeypatch):
    # prepare minimal args
    test_argv = ["predict.py", "--name", "kl8", "--seq_len", "5", "--f_data", "0"]
    monkeypatch.setattr(sys, 'argv', test_argv)

    # mock heavy work
    mod = importlib.import_module('scripts.predict')
    called = {}

    def fake_run_predict(**kwargs):
        called['ran'] = True

    monkeypatch.setattr(DEFAULT_PIPELINE, 'run_predict', fake_run_predict)

    # reload to execute top-level __main__ when invoked as script
    importlib.reload(mod)
    # if module exposes a main callable, call it to ensure args are applied
    if hasattr(mod, 'main'):
        try:
            mod.main()
        except SystemExit:
            # some scripts may call sys.exit; ignore for test
            pass

    assert DEFAULT_PIPELINE.args is not None
    assert called.get('ran', False) is True


def test_get_data_main_sets_pipeline_args(monkeypatch):
    test_argv = ["get_data.py", "--name", "kl8"]
    monkeypatch.setattr(sys, 'argv', test_argv)

    mod = importlib.import_module('scripts.get_data')

    # mock get_data_run to avoid network
    monkeypatch.setattr('src.common.get_data_run', lambda name, cq: None)

    importlib.reload(mod)
    if hasattr(mod, 'main'):
        try:
            mod.main()
        except SystemExit:
            pass

    assert DEFAULT_PIPELINE.args is not None
