"""OpenCourses.AI Colab setup helper."""
from __future__ import annotations

def prepare():
    # OpenCourses.AI Colab setup
    import importlib, json, os, pathlib, subprocess, sys, tempfile
    REPO_URL = "https://github.com/opencourses-ai-colab/opencourse-fundamentos-de-transformers-d9d23ccb.git"
    BRANCH = "main"
    VERSION = "56a5adfc10ebbfbc3a0ec4d850a4d82a48587607fc02a374e3201e8bc79bc7d5"
    EXPECTED_SOURCE_HASH = "56a5adfc10ebbfbc3a0ec4d850a4d82a48587607fc02a374e3201e8bc79bc7d5"
    REQUIRED_FILES = ["assets/audio/notebooks/01_el_problema_de_modelar_secuencias_intro.mp3","assets/audio/notebooks/02_tokens_vocabulario_y_batches_intro.mp3","assets/audio/notebooks/03_embeddings_y_geometria_de_tokens_intro.mp3","assets/audio/notebooks/04_posicion_y_orden_en_secuencias_intro.mp3","assets/audio/notebooks/05_tensores_del_transformer_intro.mp3","assets/audio/notebooks/06_atencion_como_mezcla_ponderada_intro.mp3","assets/audio/notebooks/07_queries_keys_y_values_intro.mp3","assets/audio/notebooks/08_self_attention_matricial_intro.mp3","assets/audio/notebooks/09_atencion_causal_intro.mp3","assets/audio/notebooks/10_multi_head_attention_intro.mp3","assets/audio/notebooks/11_residuales_layernorm_y_mlp_intro.mp3","assets/audio/notebooks/12_bloque_transformer_decoder_intro.mp3","assets/audio/notebooks/13_mini_gpt_desde_cero_intro.mp3","assets/audio/notebooks/14_entrenamiento_autoregresivo_intro.mp3","assets/audio/notebooks/15_generacion_interpretacion_y_limites_intro.mp3","assets/audio/notebooks/16_proyecto_final_intro.mp3","assets/audio/presentacion_curso.mp3","assets/figures/00_presentacion_del_curso.png","assets/figures/01_el_problema_de_modelar_secuencias.png","assets/figures/02_tokens_vocabulario_y_batches.png","assets/figures/03_embeddings_y_geometria_de_tokens.png","assets/figures/04_posicion_y_orden_en_secuencias.png","assets/figures/05_tensores_del_transformer.png","assets/figures/06_atencion_como_mezcla_ponderada.png","assets/figures/07_queries_keys_y_values.png","assets/figures/08_self_attention_matricial.png","assets/figures/09_atencion_causal.png","assets/figures/10_multi_head_attention.png","assets/figures/11_residuales_layernorm_y_mlp.png","assets/figures/12_bloque_transformer_decoder.png","assets/figures/13_mini_gpt_desde_cero.png","assets/figures/14_entrenamiento_autoregresivo.png","assets/figures/15_generacion_interpretacion_y_limites.png","assets/figures/16_proyecto_final.png","assets/readings/unit-10-nota-tecnica.pdf","assets/readings/unit-11-nota-tecnica.pdf","assets/readings/unit-12-nota-tecnica.pdf","assets/readings/unit-13-nota-tecnica.pdf","assets/readings/unit-14-nota-tecnica.pdf","assets/readings/unit-15-nota-tecnica.pdf","assets/readings/unit-16-nota-tecnica.pdf","assets/readings/unit-2-nota-tecnica.pdf","assets/readings/unit-3-nota-tecnica.pdf","assets/readings/unit-4-nota-tecnica.pdf","assets/readings/unit-5-nota-tecnica.pdf","assets/readings/unit-6-nota-tecnica.pdf","assets/readings/unit-7-nota-tecnica.pdf","assets/readings/unit-8-nota-tecnica.pdf","assets/readings/unit-9-nota-tecnica.pdf","notebooks/unit-1.ipynb","notebooks/unit-10.ipynb","notebooks/unit-11.ipynb","notebooks/unit-12.ipynb","notebooks/unit-13.ipynb","notebooks/unit-14.ipynb","notebooks/unit-15.ipynb","notebooks/unit-16.ipynb","notebooks/unit-17.ipynb","notebooks/unit-2.ipynb","notebooks/unit-3.ipynb","notebooks/unit-4.ipynb","notebooks/unit-5.ipynb","notebooks/unit-6.ipynb","notebooks/unit-7.ipynb","notebooks/unit-8.ipynb","notebooks/unit-9.ipynb","opencourses-colab.json","requirements.txt","src/fundamentos_transformers/__init__.py","src/fundamentos_transformers/arquitectura_experimentos.py","src/fundamentos_transformers/atencion.py","src/fundamentos_transformers/atencion_experimentos.py","src/fundamentos_transformers/componentes_interactivos.py","src/fundamentos_transformers/datos.py","src/fundamentos_transformers/embeddings.py","src/fundamentos_transformers/entrenamiento.py","src/fundamentos_transformers/entrenamiento_experimentos.py","src/fundamentos_transformers/generacion.py","src/fundamentos_transformers/generacion_experimentos.py","src/fundamentos_transformers/minigpt_experimentos.py","src/fundamentos_transformers/modelo.py","src/fundamentos_transformers/posicion.py","src/fundamentos_transformers/proyecto_final_experimentos.py","src/fundamentos_transformers/tensores.py","src/fundamentos_transformers/tokenizacion.py","src/fundamentos_transformers/visualizacion.py"]
    BASE = pathlib.Path("/content/opencourses/opencourse-fundamentos-de-transformers-d9d23ccb")
    BASE.mkdir(parents=True, exist_ok=True)
    TARGET = BASE / VERSION
    def _snapshot_issue(folder):
        if not (folder / 'notebooks').is_dir():
            return 'the notebooks folder is missing'
        missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
        if missing:
            return 'required files are missing: ' + ', '.join(missing[:5])
        if EXPECTED_SOURCE_HASH:
            try:
                manifest = json.loads((folder / 'opencourses-colab.json').read_text(encoding='utf8'))
            except (OSError, ValueError) as error:
                return f'the publication manifest cannot be read: {error}'
            if manifest.get('source_hash') != EXPECTED_SOURCE_HASH:
                return 'the publication version has changed; reopen the latest notebook from OpenCourses.AI'
        return None
    if _snapshot_issue(TARGET) is not None:
        recoveries = sorted(BASE.glob(VERSION + '-recovery-*'), key=lambda folder: folder.stat().st_mtime, reverse=True)
        existing = next((folder for folder in recoveries if _snapshot_issue(folder) is None), None)
        if existing is not None:
            TARGET = existing
        elif TARGET.exists():
            print('The course folder is incomplete. Preserving it and downloading a fresh copy.')
            TARGET = pathlib.Path(tempfile.mkdtemp(prefix=VERSION + '-recovery-', dir=BASE))
    if _snapshot_issue(TARGET) is not None:
        print('Downloading the course resources...')
        try:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(TARGET)], check=True)
        except Exception as error:
            raise RuntimeError(f'Course download failed. Files in {TARGET} were preserved. Run setup again to retry in a fresh folder.') from error
        issue = _snapshot_issue(TARGET)
        if issue is not None:
            raise RuntimeError(f'Course setup stopped: {issue}. Files in {TARGET} were preserved; no course code was started.')
    else:
        print('Using the verified folder for this course version; your local changes are preserved.')
    os.environ['COURSE_ROOT'] = str(TARGET)
    os.environ['ASSETS_DIR'] = str(TARGET / 'assets')
    src_dir = TARGET / 'src'
    if src_dir.is_dir():
        if str(src_dir) in sys.path:
            sys.path.remove(str(src_dir))
        sys.path.insert(0, str(src_dir))
    importlib.invalidate_caches()
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
