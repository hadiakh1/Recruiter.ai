#!/bin/bash

# Simple frontend startup script

cd "$(dirname "$0")/frontend"

echo "Starting frontend server..."
echo "Frontend will run on: http://localhost:3000"
echo ""

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start server
npm run dev


