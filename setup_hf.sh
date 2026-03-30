#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
if [[ -d "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

echo "Setting up Hugging Face CLI..."

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Python is not installed. Please install Python 3 first."
    exit 1
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "huggingface_hub[cli]"

echo "Verifying CLI..."
CLI_PATH="$VENV_DIR/bin/huggingface-cli"
if command -v hf >/dev/null 2>&1; then
    mkdir -p "$VENV_DIR/bin"
    cat > "$CLI_PATH" <<'EOF'
#!/bin/bash
set -e

if [[ $# -eq 0 ]]; then
    exec hf
fi

cmd="$1"
shift || true

case "$cmd" in
    login)
        exec hf auth login "$@"
        ;;
    whoami)
        exec hf auth whoami "$@"
        ;;
    logout)
        exec hf auth logout "$@"
        ;;
    *)
        exec hf "$cmd" "$@"
        ;;
esac
EOF
    chmod +x "$CLI_PATH"
    export PATH="$ROOT_DIR/$VENV_DIR/bin:$PATH"
    echo "Created compatibility shim at $CLI_PATH -> hf"
fi

if [[ -x "$CLI_PATH" ]]; then
    echo "CLI entrypoint found at $CLI_PATH"
fi

if command -v huggingface-cli &> /dev/null
then
    echo "CLI installed successfully"
    huggingface-cli whoami || echo "Please run: huggingface-cli login"
else
    echo "CLI not found. Use fallback login via git push."
fi
