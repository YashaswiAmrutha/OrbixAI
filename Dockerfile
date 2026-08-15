# OrbixAI backend (FastAPI + LangGraph + the MCP tool servers).
#
# The MCP servers are NOT separate images: agent.py spawns each one as a subprocess
# with sys.executable over stdio, so they must share this image's interpreter and
# site-packages. One image, many tool processes — matching how the host runs it.

FROM python:3.11-slim AS base

# PYTHONDONTWRITEBYTECODE: no .pyc clutter in layers.
# PYTHONUNBUFFERED: logs stream out instead of sitting in a buffer.
# PYTHONIOENCODING: the registry forces this for spawned MCP servers anyway; setting
#   it image-wide keeps non-ASCII tool output (e.g. Devanagari place names) safe.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg: faster-whisper decodes uploaded audio with it.
# git: the orbix-git MCP server shells out to the real git binary.
# curl: healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, in their own layer, so editing source doesn't reinstall them.
COPY requirements.docker.txt ./
RUN pip install -r requirements.docker.txt

# Application source. .dockerignore keeps every secret and local DB out of this.
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Non-root: the shell/filesystem MCP servers execute with this process's privileges,
# so don't hand them root. Owns /app for the SQLite cache it writes at runtime.
RUN useradd --create-home --uid 1000 orbix \
 && mkdir -p /app/backend/metrics /workspace \
 && chown -R orbix:orbix /app /workspace
USER orbix

# Container-appropriate defaults. Each is overridable from compose.
#  - Neo4j resolves by compose service name, not 127.0.0.1.
#  - Ollama likewise; its client reads OLLAMA_HOST.
#  - File/git tool roots point at the mounted workspace, NOT "/" — the host default
#    of "every fixed drive" would mean the whole container filesystem here.
ENV NEO4J_URI=neo4j://neo4j:7687 \
    OLLAMA_HOST=http://ollama:11434 \
    ORBIX_FILES_ROOTS=/workspace \
    ORBIX_GIT_ROOTS=/workspace

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8001/health || exit 1

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
