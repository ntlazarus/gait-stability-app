#!/usr/bin/env bash
set -Eeuo pipefail

# Run relative to the checkout containing this script. This can be either the
# main checkout or a linked Git worktree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop."
  exit 1
fi

if ! docker image inspect local-opencode >/dev/null 2>&1; then
  echo "Docker image 'local-opencode' does not exist."
  echo "Build the OpenCode image before running this launcher."
  exit 1
fi

if ! git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This script must be run from inside a Git repository or linked worktree."
  exit 1
fi

# OpenCode is an interactive TUI. Refuse to launch when stdin or stdout is not
# attached to a real terminal.
if [[ ! -t 0 || ! -t 1 ]]; then
  echo "OpenCode requires an interactive terminal."
  echo "Run this script directly from Terminal, iTerm, or a VS Code terminal."
  exit 1
fi

TTY_SIZE="$(stty size 2>/dev/null || true)"

if [[ -z "$TTY_SIZE" || "$TTY_SIZE" == "0 0" ]]; then
  echo "The host terminal reported an invalid size: ${TTY_SIZE:-unknown}"
  echo "Run 'reset' followed by 'stty sane', then try again."
  exit 1
fi

# The checkout OpenCode should edit:
# - Main repository when launched there
# - Feature worktree when launched from a linked worktree
HOST_WORKDIR="$(
  git -C "$SCRIPT_DIR" rev-parse --show-toplevel
)"
HOST_WORKDIR="$(cd "$HOST_WORKDIR" && pwd -P)"

# Both the main checkout and every linked worktree resolve to the same common
# Git directory: <main-repository>/.git.
GIT_COMMON_DIR="$(
  git -C "$HOST_WORKDIR" \
    rev-parse --path-format=absolute --git-common-dir
)"
GIT_COMMON_DIR="$(cd "$GIT_COMMON_DIR" && pwd -P)"

# The main repository is the parent directory of its shared .git directory.
MAIN_REPO_ROOT="$(cd "$(dirname "$GIT_COMMON_DIR")" && pwd -P)"

# Mount the main repository root so the container can access:
# - the main repository's .git/worktrees metadata;
# - all linked worktrees under .worktrees;
# - the current checkout at its exact host path.
HOST_MOUNT_ROOT="$MAIN_REPO_ROOT"

# OpenCode stores provider credentials and application state under
# ~/.local/share/opencode inside the container. Always use the main checkout's
# persistent directory so authentication is shared across worktrees.
SHARED_OPENCODE_DATA="$MAIN_REPO_ROOT/.opencode-container/auth"
mkdir -p "$SHARED_OPENCODE_DATA"

if [[ ! -f "$HOST_WORKDIR/opencode.docker.json" ]]; then
  echo "Missing Docker OpenCode configuration:"
  echo "  $HOST_WORKDIR/opencode.docker.json"
  exit 1
fi

# Generate the active project configuration for the current checkout.
cp \
  "$HOST_WORKDIR/opencode.docker.json" \
  "$HOST_WORKDIR/opencode.json"

codebase_memory_server_command="${CODEBASE_MEMORY_MCP_SERVER_COMMAND:-codebase-memory-mcp}"

if [[ -n "${CODEBASE_MEMORY_MCP_REFRESH_COMMAND:-}" ]]; then
  codebase_memory_refresh_command="$CODEBASE_MEMORY_MCP_REFRESH_COMMAND"
else
  codebase_memory_refresh_command="$(
    printf \
      "codebase-memory-mcp cli index_repository '{\"repo_path\":\"%s\"}'" \
      "$HOST_WORKDIR"
  )"
fi

# Preserve the host terminal type. macOS Terminal normally reports
# xterm-256color, but provide safe fallbacks.
HOST_TERM="${TERM:-xterm-256color}"
HOST_COLORTERM="${COLORTERM:-truecolor}"

# Optional diagnostic modes:
#
#   OPENCODE_PURE=1 ./run-opencode.sh
#       Start without external OpenCode plugins.
#
#   OPENCODE_DEBUG=1 ./run-opencode.sh
#       Print detailed OpenCode startup logs.
#
#   OPENCODE_PURE=1 OPENCODE_DEBUG=1 ./run-opencode.sh
#       Combine both modes.
opencode_args=()

case "${OPENCODE_PURE:-0}" in
  1|true|TRUE|yes|YES)
    opencode_args+=(--pure)
    ;;
esac

case "${OPENCODE_DEBUG:-0}" in
  1|true|TRUE|yes|YES)
    opencode_args+=(--print-logs --log-level DEBUG)
    ;;
esac

echo "OpenCode checkout:    $HOST_WORKDIR"
echo "Main repository:      $MAIN_REPO_ROOT"
echo "Shared OpenCode data: $SHARED_OPENCODE_DATA"
echo "Terminal:             $HOST_TERM"
echo "Terminal size:        $TTY_SIZE"

if [[ "${OPENCODE_PURE:-0}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  echo "Plugin mode:          pure — external plugins disabled"
else
  echo "Plugin mode:          normal"
fi

# This launcher intentionally supports one active OpenCode container at a time.
docker rm -f opencode-sandbox >/dev/null 2>&1 || true

docker_args=(
  --rm
  --interactive
  --tty
  --init

  --name opencode-sandbox

  --cap-drop=ALL
  --security-opt no-new-privileges

  # Preserve identical absolute paths inside and outside the container. This is
  # required for linked worktree .git metadata to resolve correctly.
  --volume "$HOST_MOUNT_ROOT:$HOST_MOUNT_ROOT"

  # Share provider authentication and OpenCode state across every worktree.
  --volume "$SHARED_OPENCODE_DATA:/root/.local/share/opencode"

  # Explicitly provide terminal capabilities to the OpenCode TUI.
  --env "TERM=$HOST_TERM"
  --env "COLORTERM=$HOST_COLORTERM"
  --env "OPENCODE_DISABLE_TERMINAL_TITLE=true"

  # Prevent OpenCode itself from changing versions unexpectedly inside this
  # controlled container environment.
  --env "OPENCODE_DISABLE_AUTOUPDATE=true"

  --env "CODEBASE_MEMORY_MCP_SERVER_COMMAND=$codebase_memory_server_command"
  --env "CODEBASE_MEMORY_MCP_REFRESH_COMMAND=$codebase_memory_refresh_command"
  --env "CODEBASE_MEMORY_IDLE_SECONDS=${CODEBASE_MEMORY_IDLE_SECONDS:-300}"
  --env "CODEBASE_MEMORY_COMMAND_TIMEOUT_SECONDS=${CODEBASE_MEMORY_COMMAND_TIMEOUT_SECONDS:-60}"
  --env "CODEBASE_MEMORY_DIAGRAM_IDLE_SECONDS=${CODEBASE_MEMORY_DIAGRAM_IDLE_SECONDS:-1200}"

  # OpenCode edits the checkout from which this launcher was started.
  --workdir "$HOST_WORKDIR"

  # Do not rely on the image's default ENTRYPOINT or CMD.
  --entrypoint opencode
)

# Replace this shell with the Docker CLI process. This gives Docker direct
# ownership of stdin, stdout, signals, and the terminal session.
if [[ "${#opencode_args[@]}" -gt 0 ]]; then
  exec docker run \
    "${docker_args[@]}" \
    local-opencode \
    "${opencode_args[@]}"
else
  exec docker run \
    "${docker_args[@]}" \
    local-opencode
fi
