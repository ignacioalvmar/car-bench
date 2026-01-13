import os
import sys
import traceback

from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
    ContextState,
)
from car_bench.envs.car_voice_assistant.context.fixed_context import FixedContext

from .data_manager import DataManager

# Define lists (you can choose one list or separate ones)
ALL_STATE_MODELS = [ContextState]

ALL_CONTEXT_MODELS = [FixedContext]

ALL_MODELS = ALL_STATE_MODELS + ALL_CONTEXT_MODELS


print("Initializing shared Car VA DataManager...")

_data_directory_path = os.getenv("CAR_BENCH_DATA_DIR", os.path.dirname(__file__))

try:
    # Instantiate the DataManager using the determined path
    car_va_data_manager = DataManager(base_data_path=_data_directory_path, preload=True)
    print(
        f"Shared Car VA DataManager instance created successfully. Data path: {_data_directory_path}"
    )

    # Optional: Trigger eager loading of caches here if preferred over lazy loading
    # print("Pre-loading caches...")
    # _ = car_va_data_manager.locations
    # _ = car_va_data_manager.pois
    # _ = car_va_data_manager.weather
    # _ = car_va_data_manager.contacts
    # print("Caches pre-loaded.")

except Exception as e:
    print(
        f"FATAL ERROR: Could not initialize shared Car VA DataManager.", file=sys.stderr
    )
    print(f"Base data path used: {_data_directory_path}", file=sys.stderr)
    print(f"Error Type: {type(e).__name__}", file=sys.stderr)
    print(f"Error Message: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    # Define the variable as None so imports don't break, but checks will fail
    car_va_data_manager = None
    # Optionally, re-raise the exception if the application cannot proceed without the manager
    # raise RuntimeError("Failed to initialize critical DataManager") from e

# Make the instance easily importable from the 'data' package
__all__ = [
    "car_va_data_manager",
    "DataManager",
]  # Expose both the instance and the class type

# TODO: Remove
import json
import os
from typing import Any

FOLDER_PATH = os.path.dirname(__file__)
# CURRENT_LOCATION = "Munoria"


def load_data() -> dict[str, Any]:
    return {}
