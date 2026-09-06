"""OpenCourses.AI Colab setup helper."""
from __future__ import annotations

def prepare():
    # OpenCourses.AI Colab setup
    import os, pathlib, subprocess, sys
    REPO_URL = "https://github.com/opencourses-ai-colab/opencourse-fundamentos-de-transformers-d9d23ccb.git"
    BRANCH = "main"
    TARGET = pathlib.Path("/content/opencourses/opencourse-fundamentos-de-transformers-d9d23ccb")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        print('Downloading the course resources...')
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(TARGET)], check=True)
    else:
        print('Using the existing course folder; your local changes are preserved.')
    os.environ['COURSE_ROOT'] = str(TARGET)
    os.environ['ASSETS_DIR'] = str(TARGET / 'assets')
    src_dir = TARGET / 'src'
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    notebooks_dir = TARGET / 'notebooks'
    if notebooks_dir.exists():
        os.chdir(notebooks_dir)
    requirements = TARGET / 'requirements.txt'
    if requirements.exists() and requirements.read_text(encoding='utf8').strip():
        print('Checking the course Python dependencies...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], check=True)
    try:
        from google.colab import output as colab_output
    except ImportError:
        pass
    else:
        try:
            colab_output.enable_custom_widget_manager()
        except Exception as error:
            print(f'Widget support could not be enabled: {error}')
    print(f'COURSE_ROOT={TARGET}')
    print('Setup complete. Run the remaining notebook cells from top to bottom.')
    return TARGET

if __name__ == '__main__':
    prepare()
