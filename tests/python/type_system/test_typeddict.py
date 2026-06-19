import unittest
from typing import TypedDict
from lirien import verify, i64


class Config(TypedDict):
    id: i64
    timeout: i64
    enabled: bool


@verify
def configure(cfg: Config) -> i64:
    # Lirien compiles this to: load.i64 (base_ptr + 8)
    # The string keys ("timeout") are completely compiled away!
    if cfg["enabled"]:
        return cfg["timeout"]
    return 0


@verify
def update_config(cfg: Config, new_timeout: i64) -> None:
    cfg["timeout"] = new_timeout


class TestTypedDict(unittest.TestCase):
    def test_basic_typeddict(self):
        # In Python, we pass a dictionary
        cfg = {"id": 1, "timeout": 1000, "enabled": True}
        res = configure(cfg)
        self.assertEqual(res, 1000)

        cfg2 = {"id": 2, "timeout": 5000, "enabled": False}
        res2 = configure(cfg2)
        self.assertEqual(res2, 0)

    def test_update_typeddict(self):
        cfg = {"id": 1, "timeout": 1000, "enabled": True}
        update_config(cfg, 2000)
        self.assertEqual(cfg["timeout"], 2000)


if __name__ == "__main__":
    unittest.main()
