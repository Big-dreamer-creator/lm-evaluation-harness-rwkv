from unittest.mock import patch

from lm_eval.api.model import CachingLM


class DummyLM:
    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook


def test_caching_lm_disables_sqlitedict_outer_stack(tmp_path):
    cache_path = tmp_path / "responses.db"

    with patch("sqlitedict.SqliteDict") as sqlite_dict:
        CachingLM(DummyLM(), str(cache_path))

    sqlite_dict.assert_called_once_with(
        str(cache_path), autocommit=True, outer_stack=False
    )
