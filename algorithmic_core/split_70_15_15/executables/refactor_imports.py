import re
from pathlib import Path

files = [
    "generate_main_results.py",
    "locality_analysis.py",
    "multistep_averaged.py",
    "split_study.py"
]

for filename in files:
    content = Path(filename).read_text()
    content = re.sub(r'from pipeline import', r'from core.pipeline import', content)
    content = re.sub(r'from models import', r'from core.models import', content)
    content = re.sub(r'from train import', r'from core.train import', content)
    content = re.sub(r'from model_cache import', r'from core.model_cache import', content)
    Path(filename).write_text(content)
    print(f"Updated imports in {filename}")
