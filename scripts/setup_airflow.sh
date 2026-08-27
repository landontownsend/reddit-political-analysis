#!/usr/bin/env bash
#
# Set up a dedicated virtual environment for the Airflow-orchestrated pipeline
# (dags/reddit_political_analysis_dag.py) and initialise Airflow locally.
#
# Usage:
#   ./scripts/setup_airflow.sh
#   source airflow_venv/bin/activate
#   export AIRFLOW_HOME=~/airflow REDDIT_PROJECT_ROOT="$(pwd)"
#   airflow standalone
#   cp dags/reddit_political_analysis_dag.py "$AIRFLOW_HOME/dags/"
#   airflow dags trigger reddit_political_analysis
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
AIRFLOW_VERSION="${AIRFLOW_VERSION:-2.10.4}"
PYVER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINTS="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYVER}.txt"

echo ">> Creating airflow_venv with $PYTHON_BIN ($PYVER)"
"$PYTHON_BIN" -m venv airflow_venv
./airflow_venv/bin/pip install --upgrade pip setuptools wheel

echo ">> Installing apache-airflow $AIRFLOW_VERSION (with official constraints)"
./airflow_venv/bin/pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "$CONSTRAINTS"

echo ">> Installing pipeline dependencies"
./airflow_venv/bin/pip install -r requirements-airflow.txt

echo ">> Downloading NLTK data (vader_lexicon, stopwords)"
./airflow_venv/bin/python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"

# macOS workaround: Airflow's forking task runner calls setproctitle() in the
# forked child; on macOS that reaches into CoreFoundation (not fork-safe) and the
# task hangs forever in 'running'. Make setproctitle a no-op in this venv.
if [[ "$(uname)" == "Darwin" ]]; then
  SITE_DIR="$(./airflow_venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  cat > "$SITE_DIR/zzz_darwin_setproctitle_fix.pth" <<'PTH'
import sys; exec("try:\n import setproctitle as _s\n (setattr(_s,'setproctitle',lambda *a,**k:None), setattr(_s,'setthreadtitle',lambda *a,**k:None)) if sys.platform=='darwin' else None\nexcept Exception:\n pass")
PTH
  echo ">> Installed macOS setproctitle no-op hook at $SITE_DIR/zzz_darwin_setproctitle_fix.pth"
fi

cat <<'DONE'

Done. Next steps:

  source airflow_venv/bin/activate
  export AIRFLOW_HOME=~/airflow
  export REDDIT_PROJECT_ROOT="$(pwd)"
  airflow standalone           # first run prints an admin password

  # in another shell (same env):
  cp dags/reddit_political_analysis_dag.py "$AIRFLOW_HOME/dags/"
  airflow dags trigger reddit_political_analysis

On macOS, if the webserver's log workers crash, run the components separately:
  airflow scheduler --skip-serve-logs &
  airflow webserver --port 8080 &
DONE
