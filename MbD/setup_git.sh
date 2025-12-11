#!/bin/bash
# Script to initialize git repository and prepare for GitHub

echo "=========================================="
echo "Setting up Git repository for NNV project"
echo "=========================================="

# Initialize git repository
echo "1. Initializing git repository..."
git init

# Add all files
echo "2. Adding files to git..."
git add .

# Create initial commit
echo "3. Creating initial commit..."
git commit -m "Initial commit: Battery RUL prediction with MbD model and formal verification"

echo ""
echo "=========================================="
echo "Git repository initialized successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Go to GitHub and create a new repository (don't initialize with README)"
echo "2. Copy the repository URL (e.g., https://github.com/username/repo-name.git)"
echo "3. Run the following commands:"
echo ""
echo "   git remote add origin <YOUR_REPO_URL>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""

