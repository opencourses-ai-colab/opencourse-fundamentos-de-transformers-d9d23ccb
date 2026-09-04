"""OpenCourses.AI Colab setup helper."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPO_URL = "https://github.com/opencourses-ai-colab/opencourse-fundamentos-de-transformers-d9d23ccb.git"
BRANCH = "main"
TARGET = pathlib.Path("/content/opencourses/opencourse-fundamentos-de-transformers-d9d23ccb")

def prepare():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(TARGET)], check=True)
    os.environ['COURSE_ROOT'] = str(TARGET)
    os.environ['ASSETS_DIR'] = str(TARGET / 'assets')
    notebooks_dir = TARGET / 'notebooks'
    if notebooks_dir.exists():
        os.chdir(notebooks_dir)
    requirements = TARGET / 'requirements.txt'
    if requirements.exists() and requirements.read_text(encoding='utf8').strip():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], check=True)
    return TARGET

if __name__ == '__main__':
    root = prepare()
    print(f'COURSE_ROOT={root}')
