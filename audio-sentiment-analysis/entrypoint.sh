#!/bin/sh
set -e
MODEL_REPO="${MODEL_REPO:-qetzi/voice-emotion-models}"
if [ ! -f models_final/.ready ] && [ ! -d models_final/audio ]; then
  echo "Downloading weights from $MODEL_REPO ..."
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='models_final', local_dir_use_symlinks=False)"
  touch models_final/.ready
fi
exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-7860}"
