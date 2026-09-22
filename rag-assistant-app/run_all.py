from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK = NOTEBOOK_DIR / "rag_pipeline.ipynb"
REPORT_NAME = "rag_pipeline_output"  # -> notebooks/rag_pipeline_output.html
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8501"
PY = sys.executable

# import name -> pip package name
PACKAGES = {
    "nbconvert": "nbconvert", "ipykernel": "ipykernel", "pandas": "pandas", "pypdf": "pypdf",
    "chromadb": "chromadb", "sentence_transformers": "sentence-transformers", "ollama": "ollama",
    "fastapi": "fastapi", "pydantic_settings": "pydantic-settings",
    "uvicorn": "uvicorn", "streamlit": "streamlit", "requests": "requests", "dotenv": "python-dotenv",
    "python_multipart": "python-multipart",
}
# some packages can be imported under an older name
ALTERNATIVE_IMPORTS = {"python_multipart": ["multipart"]}
NEEDS = {
    "notebook": ["nbconvert", "ipykernel", "pandas", "pypdf", "chromadb", "sentence_transformers", "ollama"],
    "serve": ["fastapi", "uvicorn", "pydantic_settings", "chromadb", "sentence_transformers", "ollama",
              "streamlit", "requests", "dotenv", "python_multipart", "pypdf"],
}


def banner(text: str) -> None:
    print("\n" + "=" * 72 + f"\n {text}\n" + "=" * 72, flush=True)


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def ensure_env_files() -> None:
    for folder in (BACKEND, FRONTEND):
        example, env = folder / ".env.example", folder / ".env"
        if example.exists() and not env.exists():
            shutil.copy(example, env)
            print(f"Created {env.relative_to(ROOT)} from .env.example")


def missing_packages(steps: list[str]) -> list[str]:
    needed = {mod for step in steps for mod in NEEDS[step]}
    def installed(module: str) -> bool:
        return any(importlib.util.find_spec(m) is not None for m in [module, *ALTERNATIVE_IMPORTS.get(module, [])])

    return sorted(PACKAGES[m] for m in needed if not installed(m))


def check_ollama(host: str, model: str) -> str | None:
    """Return an error message, or None if Ollama is up and the model is installed."""
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5) as response:
            installed = [m["name"] for m in json.load(response).get("models", [])]
    except Exception:
        return f"Cannot reach Ollama at {host}. Start the Ollama app/server, then run this again."
    if not any(name == model or name.split(":")[0] == model for name in installed):
        return f"Model '{model}' is not installed. Run:  ollama pull {model}"
    return None


def clean_stream(text: str) -> str:
    # progress bars rewrite the same line with carriage returns: keep only its final state
    text = text.replace("\r\n", "\n")  # real Windows line endings first
    return "\n".join(line.split("\r")[-1] for line in text.split("\n")).strip("\n")


def print_outputs(notebook_path: Path) -> int:
    """Print only the outputs of an executed notebook, grouped by section."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    section, printed = "", 0
    for cell in nb["cells"]:
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            heading = next((l for l in source.splitlines() if l.startswith("#")), None)
            if heading:
                section = heading.lstrip("# ").strip()
            continue
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                text = out["text"]
            elif kind in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
            elif kind == "error":
                text = "\n".join(out.get("traceback", []))
            else:
                continue
            text = clean_stream(text if isinstance(text, str) else "".join(text))
            if not text:
                continue
            if section:
                print(f"\n--- {section} ---")
                section = ""
            print(text)
            printed += 1
    return printed


def notebook_env() -> dict[str, str]:
    """Environment for the notebook: OLLAMA_HOST / OLLAMA_MODEL from backend/.env unless already set."""
    env = dict(os.environ)
    for key, value in read_env_file(BACKEND / ".env").items():
        if key in ("OLLAMA_HOST", "OLLAMA_MODEL") and key not in os.environ:
            env[key] = value
    return env


def run_notebook() -> None:
    banner("1/2  Running the notebook (top to bottom)")
    run = lambda cmd: subprocess.run(cmd, cwd=NOTEBOOK_DIR, check=True, env=notebook_env())
    try:
        # execute once, saving the outputs into the notebook (nice to see on GitHub too)
        run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=3600", str(NOTEBOOK)])
        # output-only report: code cells hidden
        run([PY, "-m", "jupyter", "nbconvert", "--to", "html", "--no-input",
             "--output", REPORT_NAME, "--output-dir", str(NOTEBOOK_DIR), str(NOTEBOOK)])
    except subprocess.CalledProcessError:
        fail("The notebook failed. Scroll up to see which cell raised the error.")
    banner("Notebook output (code hidden)")
    print_outputs(NOTEBOOK)
    print(f"\nOutput-only report saved to: {NOTEBOOK_DIR / (REPORT_NAME + '.html')}")
    print(f"Evaluation table saved to:   {NOTEBOOK_DIR / 'evaluation_results.csv'}")


def wait_for(url: str, timeout: int, process: subprocess.Popen) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and process.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1.5)
    return False


def serve() -> None:
    banner("2/2  Starting the backend and the frontend")
    ensure_env_files()
    if not (BACKEND / "data" / "vector_store" / "config.json").exists():
        fail("The vector store is missing. Run without --skip-notebook first.")
    procs: list[subprocess.Popen] = []
    try:
        backend = subprocess.Popen([PY, "-m", "uvicorn", "app.main:app", "--port", "8000"], cwd=BACKEND)
        procs.append(backend)
        print("Waiting for the backend (it loads the models; the first start can take a minute)...")
        if not wait_for(f"{BACKEND_URL}/health", 300, backend):
            fail("The backend did not start. Check the messages above.")
        env = {**os.environ, "API_BASE_URL": os.environ.get("API_BASE_URL", BACKEND_URL)}
        frontend = subprocess.Popen(
            [PY, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
             "--server.port", "8501", "--browser.gatherUsageStats", "false"],
            cwd=FRONTEND, env=env,
        )
        procs.append(frontend)
        time.sleep(4)
        webbrowser.open(FRONTEND_URL)
        print(f"\nBackend : {BACKEND_URL}/docs\nFrontend: {FRONTEND_URL}\nPress Ctrl+C to stop both.")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("A process exited, shutting down.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def read_requirements() -> list[str]:
    """Collect every pinned requirement from the three requirements files."""
    specs: list[str] = []
    for path in (ROOT / "requirements-notebook.txt", BACKEND / "requirements.txt", FRONTEND / "requirements.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")) and line not in specs:
                specs.append(line)
    return specs


def install_all() -> None:
    banner("Installing dependencies")
    bits = 8 * __import__("struct").calcsize("P")
    print(f"Python {sys.version.split()[0]} ({bits}-bit) at {PY}")
    if sys.version_info >= (3, 13):
        print("WARNING: Python 3.13+ may have no ready-made builds for some packages. Python 3.12 is recommended.")
    if bits != 64:
        print("WARNING: 32-bit Python cannot install PyTorch. Install the 64-bit version of Python 3.12.")
    subprocess.run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    if subprocess.run([PY, "-m", "pip", "install", "-r", str(ROOT / "requirements-all.txt")]).returncode == 0:
        importlib.invalidate_caches()
        print("\nAll packages installed.")
        return

    print("\nThe one-shot install failed. Retrying package by package to find the culprit...\n")
    failed = [spec for spec in read_requirements() if subprocess.run([PY, "-m", "pip", "install", spec]).returncode != 0]
    importlib.invalidate_caches()
    if failed:
        fail(
            "Could not install: " + ", ".join(failed) + "\n\n"
            "Common causes:\n"
            "  - 'Microsoft Visual C++ 14.0 or greater is required' (chromadb): install the Microsoft C++ Build Tools\n"
            "    (visualstudio.microsoft.com/visual-cpp-build-tools), or use Python 3.12 which has ready-made builds.\n"
            "  - 'No matching distribution found' (torch / sentence-transformers): your Python is too new or 32-bit.\n"
            "    Install 64-bit Python 3.12 and create a fresh virtual environment.\n"
            "  - Network errors: check your internet connection or VPN, then run again."
        )
    print("\nAll packages installed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole RAG project with one command.")
    parser.add_argument("--install", action="store_true", help="pip install all dependencies first")
    parser.add_argument("--skip-notebook", action="store_true", help="do not run the notebook")
    parser.add_argument("--no-serve", action="store_true", help="do not start the backend and frontend")
    args = parser.parse_args()

    if args.install:
        install_all()

    steps = [s for s, skip in (("notebook", args.skip_notebook), ("serve", args.no_serve)) if not skip]
    if not steps:
        print("Nothing to do.")
        return

    missing = missing_packages(steps)
    if missing:
        fail("Missing packages: " + ", ".join(missing) + "\nThey are not installed in THIS Python: " + PY +
             "\nInstall everything with:  python run_all.py --install")

    if "notebook" in steps or "serve" in steps:
        cfg = {**read_env_file(BACKEND / ".env.example"), **read_env_file(BACKEND / ".env"), **os.environ}
        error = check_ollama(cfg.get("OLLAMA_HOST", "http://localhost:11434"), cfg.get("OLLAMA_MODEL", "llama3.2"))
        if error:
            fail(error)

    if "notebook" in steps:
        run_notebook()
    if "serve" in steps:
        serve()


if __name__ == "__main__":
    main()
