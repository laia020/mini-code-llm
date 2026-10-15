"""Prepare the local Python environment and open Mini Code LLM."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
PACKAGES = ("torch", "transformers", "gradio", "huggingface_hub")


def environment_python(directory):
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def ensure_environment(root):
    directory = root / ".venv"
    python = environment_python(directory)
    if not directory.exists():
        print("Preparando o ambiente do projeto...", flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
    if not python.is_file():
        raise RuntimeError("A pasta .venv existente está incompleta. Preserve seus arquivos e use uma cópia limpa do projeto.")
    version = subprocess.check_output(
        [str(python), "-c", "import sys; print('%s.%s' % sys.version_info[:2])"], text=True,
    ).strip()
    if version != "3.11":
        raise RuntimeError("A .venv existente não usa Python 3.11. Use uma cópia limpa do projeto com Python 3.11.")
    return python


def installed_versions(python):
    code = (
        "import importlib.metadata as m, json; "
        f"print(json.dumps({{name:m.version(name) for name in {PACKAGES!r}}}))"
    )
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    if result.returncode:
        return None
    return json.loads(result.stdout)


def compatible_torch(python):
    result = subprocess.run(
        [str(python), "-c", "import importlib.metadata as m; print(m.version('torch'))"],
        capture_output=True, text=True,
    )
    match = re.match(r"(\d+)\.(\d+)", result.stdout.strip())
    return result.returncode == 0 and match is not None and tuple(map(int, match.groups())) >= (2, 5)


def ensure_dependencies(root, python):
    requirements = root / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = root / ".venv" / "mini-code-llm-setup.json"
    current = installed_versions(python)
    try:
        previous = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    if current and previous == {"requirements_sha256": digest, "versions": current}:
        print("Ambiente pronto. Reutilizando as dependências instaladas.", flush=True)
        return
    # CPU is enough for the demo. Preserve an existing compatible CUDA installation.
    if sys.platform in {"win32", "linux"} and not compatible_torch(python):
        print("Instalando PyTorch para demonstração em CPU...", flush=True)
        subprocess.run([
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "torch>=2.5", "--index-url", "https://download.pytorch.org/whl/cpu",
        ], cwd=root, check=True)
    print("Instalando ou conferindo as dependências. A primeira execução pode demorar...", flush=True)
    subprocess.run([
        str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements),
    ], cwd=root, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], cwd=root, check=True)
    versions = installed_versions(python)
    if not versions:
        raise RuntimeError("Não foi possível verificar a instalação das dependências.")
    # Only mark successful installations; an interrupted setup can be retried.
    stamp.write_text(json.dumps({"requirements_sha256": digest, "versions": versions}) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Prepara o ambiente sem iniciar o frontend.")
    parser.add_argument("--no-browser", action="store_true", help="Inicia sem abrir o navegador automaticamente.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Instale Python 3.11 para executar este inicializador. https://www.python.org/downloads/")
        if not 1 <= args.port <= 65535:
            raise ValueError("A porta deve estar entre 1 e 65535.")
        python = ensure_environment(ROOT)
        ensure_dependencies(ROOT, python)
        if args.prepare_only:
            print("Instalação concluída. Execute iniciar.bat ou python iniciar.py para abrir o aplicativo.", flush=True)
            return 0
        print(f"Abrindo Mini Code LLM em http://127.0.0.1:{args.port}", flush=True)
        print("Mantenha esta janela aberta enquanto usa o aplicativo. Ctrl+C encerra o servidor.", flush=True)
        command = [str(python), str(ROOT / "src/app.py"), "--port", str(args.port)]
        if not args.no_browser:
            command.append("--open-browser")
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        print("\nAplicativo encerrado.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Não foi possível iniciar: {error}", file=sys.stderr)
        print("Confira o Python 3.11 e a conexão com a internet e execute novamente.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
