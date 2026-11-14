#!/bin/bash
# Deploy script for the Clinical Reasoning Agent's Hugging Face Space.
# Mirrors hf_space/deploy.sh's pattern, but also copies the real
# agents/ehr/serving.healthcare source (not a duplicated port) into the
# Space clone, since app.py imports them directly.

set -e

SPACE_NAME="clinical-reasoning-agent-demo"
HF_USER="${HF_USER:-your-username}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Deploying Clinical Reasoning Agent demo to Hugging Face Spaces..."
echo "Space: $HF_USER/$SPACE_NAME"

if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing Hugging Face CLI..."
    pip install -U huggingface_hub[cli]
fi

if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to Hugging Face:"
    huggingface-cli login
fi

echo "Creating Space $HF_USER/$SPACE_NAME..."
huggingface-cli repo create "$HF_USER/$SPACE_NAME" --type space --space-sdk docker --private || true

cd /tmp
rm -rf "$SPACE_NAME"
git clone "https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"
cd "$SPACE_NAME"

# Copy the Space's own files, then the real source it imports.
cp -r "$REPO_ROOT/hf_space_agent/"* .
mkdir -p agents ehr serving/healthcare
cp -r "$REPO_ROOT/agents/"* agents/
cp -r "$REPO_ROOT/ehr/"* ehr/
cp "$REPO_ROOT/serving/__init__.py" serving/__init__.py
cp -r "$REPO_ROOT/serving/healthcare/"* serving/healthcare/

git config user.email "deploy@glucopulse.local"
git config user.name "GlucoPulse Deploy"

git add .
git commit -m "Deploy Clinical Reasoning Agent demo" || echo "No changes to commit"
git push origin main

echo "Deployment complete! Space available at:"
echo "https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"
echo "Demo page at: https://$HF_USER-$SPACE_NAME.hf.space/"
