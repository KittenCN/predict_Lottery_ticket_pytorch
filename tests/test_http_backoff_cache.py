import os
import sys
import tempfile
import types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.common import get_http_session_with_backoff


def test_http_backoff_uses_cache(monkeypatch, tmp_path):
    # prepare a fake cache dir and file
    from src.config import HTTP_CACHE_DIR as orig_cache
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    try:
        # monkeypatch config module's HTTP_CACHE_DIR
        import importlib
        cfg = importlib.import_module('src.config')
        cfg.HTTP_CACHE_DIR = str(cache_dir)

        url = 'https://example.invalid/test'
        import hashlib
        key = hashlib.sha256(url.encode('utf-8')).hexdigest()
        cache_path = cache_dir / f"{key}.cache"
        cache_text = 'CACHED_CONTENT'
        cache_path.write_text(cache_text, encoding='utf-8')

        # monkeypatch requests.Session.get to always raise to force fallback
        class DummyRespException(Exception):
            pass

        def fake_get(self, *args, **kwargs):
            raise DummyRespException('network down')

        import requests
        monkeypatch.setattr(requests.Session, 'get', fake_get)

        # include allowlist so function will proceed to attempt network and then fallback to cache
        resp = get_http_session_with_backoff(url, retries=2, backoff_base=0.01, timeout=0.1, allowlist=['example.invalid'])
        assert hasattr(resp, 'text')
        assert 'CACHED_CONTENT' in resp.text
    finally:
        # restore original config if needed
        try:
            cfg.HTTP_CACHE_DIR = orig_cache
        except Exception:
            pass
