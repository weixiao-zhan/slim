import glob
import os


def get_nvidia_ld_library_path() -> dict[str, str]:
    """Auto-detect nvidia lib paths from pip-installed packages for LD_LIBRARY_PATH.

    Returns a dict suitable for unpacking into Ray runtime_env env_vars.
    """
    try:
        nvidia_base = os.path.join(os.path.dirname(os.path.dirname(__import__("nvidia").__file__)), "nvidia")
        lib_dirs = glob.glob(os.path.join(nvidia_base, "*/lib"))
        if lib_dirs:
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            return {"LD_LIBRARY_PATH": ":".join(lib_dirs) + (f":{existing}" if existing else "")}
    except (ImportError, Exception):
        pass
    return {}
