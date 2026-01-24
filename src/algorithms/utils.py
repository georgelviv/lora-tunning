from pathlib import Path
import shutil
from typing import Dict, Optional


def current_limit_for_tp(tp: int) -> int:
  if tp <= 7:
    return 80
  elif tp <= 13:
    return 100
  elif tp <= 17:
    return 120
  else:
    return 140

def prepare_results(
  base_dir: Path,
  extra_files: Optional[Dict[str, str]] = None
) -> Dict[str, Path]:
  results_dir = base_dir / "results"

  if results_dir.exists() and results_dir.is_dir():
    shutil.rmtree(results_dir)

  results_dir.mkdir()

  paths: Dict[str, Path] = {
    "results": results_dir / "results.csv",
    "history": results_dir / "history.csv",
  }

  if extra_files:
      for key, filename in extra_files.items():
          paths[key] = results_dir / filename

  return paths