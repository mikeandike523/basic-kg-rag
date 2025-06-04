#!/bin/bash

# Ensure one argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

INPUT_FILE="$1"
DEST_FILE="./story_learner/sectionizer-viewer/sections.json"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file '$INPUT_FILE' not found."
  exit 1
fi

# Copy input file to destination
cp "$INPUT_FILE" "$DEST_FILE"
echo "Copied '$INPUT_FILE' to '$DEST_FILE'."

# Find first available port between 3001 and 3999
for PORT in $(seq 3001 3999); do
  if ! lsof -i:$PORT >/dev/null; then
    FOUND_PORT=$PORT
    break
  fi
done

if [ -z "$FOUND_PORT" ]; then
  echo "Error: No available port found in the range 3001-3999."
  exit 1
fi

echo "Starting http-server on port $FOUND_PORT..."
# Run the server

nvm use node
pnpx http-server ./story_learner/sectionizer-viewer -p "$FOUND_PORT"
