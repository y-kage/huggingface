import importlib
import os

current_dir = os.path.dirname(__file__)

# __all__ 変数を設定して、公開するモジュールを指定
__all__ = [
    filename[:-3]
    for filename in os.listdir(current_dir)
    if filename.endswith(".py")
    and filename != "__init__.py"
    and not filename.startswith("_")
]

# すべてのモジュールをインポート
for module_name in __all__:
    importlib.import_module(f".{module_name}", package=__name__)
