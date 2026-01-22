from pathlib import Path
import shutil

def prepare_results(base_dir: Path):
  results_dir = base_dir / "results"

  if results_dir.exists() and results_dir.is_dir():
    shutil.rmtree(results_dir)

  results_dir.mkdir()

  results_file = results_dir / "results.csv"
  history_file = results_dir / "history.csv"

  # results_file.touch()
  # history_file.touch()

  return results_file, history_file