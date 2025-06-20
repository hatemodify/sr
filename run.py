import os
import platform
import subprocess
import sys

VENV_DIR = "venv"
PYTHON_EXE = sys.executable

def is_windows():
    return platform.system() == "Windows"

def venv_python():
    return os.path.join(VENV_DIR, "Scripts" if is_windows() else "bin", "python")

def run_command(cmd, shell=False):
    print(f"🛠️  Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, shell=shell, check=True)

def main():
    if not os.path.isdir(VENV_DIR):
        print("🔧 Creating virtual environment...")
        run_command([PYTHON_EXE, "-m", "venv", VENV_DIR])

    print("📦 Installing required packages...")
    run_command([venv_python(), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([
        venv_python(), "-m", "pip", "install",
        "setuptools",
        "numpy==1.26.4",            # ✅ pandas_ta 호환
        "pandas",
        "pandas_ta",
        "requests",
        "beautifulsoup4",
        "lxml",
        "scikit-learn",              # ✅ MinMaxScaler 에 필요
        "html5lib",
        "httpx",
        "asyncio"
    ])

    print("📝 Saving requirements.txt...")
    with open("requirements.txt", "w") as f:
        subprocess.run([venv_python(), "-m", "pip", "freeze"], stdout=f)

    print("🚀 Running ranking_score.py...")
    run_command([venv_python(), "ranking_score.py"])

if __name__ == "__main__":
    main()
