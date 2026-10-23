"""L16 Web Lab core. Verze = `backend/pyproject.toml` (frontend a native ji musí mít stejnou)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("l16-web-lab")
except PackageNotFoundError:  # zdrojáky bez instalace
    __version__ = "0.0.0+dev"
