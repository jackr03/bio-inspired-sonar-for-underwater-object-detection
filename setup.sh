#!/usr/bin/env bash  

# Activate Virtual Environment
echo 'Activating virtual environment...'
source .venv/bin/activate
export PYTHONPATH="$(pwd)/.venv/lib/python3.12/site-packages:$(pwd):$PYTHONPATH"

# Set up SSH Key for GitHub
echo 'Starting fresh SSH agent...'
eval "$(ssh-agent -s)"

echo 'Adding SSH key...'
ssh-add ~/.ssh/id_ed25519

echo 'Testing GitHub SSH connection...'
ssh -T git@github.com

echo 'Interactive session ready.'