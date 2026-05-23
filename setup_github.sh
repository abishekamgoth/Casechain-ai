#!/bin/bash
# ============================================================
# setup_github.sh
# ============================================================
# RUN THIS to push your project to GitHub TODAY.
# Make sure you've created the GitHub repo first!
#
# Steps:
# 1. Create a new repo on github.com named "legal-rag-project"
#    (go to github.com → New repository → name it → Create)
# 2. Replace YOUR_GITHUB_USERNAME below with your actual username
# 3. Run: bash setup_github.sh
# ============================================================

# ---- CONFIG: Change this! ----
GITHUB_USERNAME="abishekgamoth"       # ← Your GitHub username
REPO_NAME="legal-rag-project"
BRANCH="main"

echo "============================================"
echo "  Pushing Legal RAG Project to GitHub"
echo "============================================"
echo ""

# Step 1: Initialize git repo
echo "[1/6] Initializing git repository..."
git init
echo "✅ Git initialized"

# Step 2: Add all files (respects .gitignore)
echo "[2/6] Staging all files..."
git add .
echo "✅ Files staged"

# Step 3: First commit
echo "[3/6] Creating initial commit..."
git commit -m "🚀 Initial commit: Legal Case Prediction RAG System

- LangChain RetrievalQA pipeline with custom legal prompt
- FAISS vector store for semantic case retrieval  
- NetworkX citation graph for precedent chain discovery
- FastAPI REST API with /predict, /search, /citation-graph endpoints
- MLflow experiment tracking for all predictions
- Docker + docker-compose for containerized deployment
- GitHub Actions CI with accuracy gate
- Comprehensive unit tests for all components
- Full documentation and interview guide

Datasets: ECHR, SCOTUS, Indian Kanoon
Tech stack: Python, LangChain, FAISS, NetworkX, FastAPI, MLflow, Docker"

echo "✅ Commit created"

# Step 4: Add remote origin
echo "[4/6] Adding GitHub remote..."
git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
echo "✅ Remote added: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# Step 5: Set main branch
echo "[5/6] Setting branch to main..."
git branch -M ${BRANCH}
echo "✅ Branch set to: ${BRANCH}"

# Step 6: Push to GitHub
echo "[6/6] Pushing to GitHub..."
echo ""
echo "You may be asked for your GitHub credentials."
echo "Use your GitHub username and a Personal Access Token (not your password)."
echo "Create a token at: https://github.com/settings/tokens"
echo ""
git push -u origin ${BRANCH}

echo ""
echo "============================================"
echo "  ✅ SUCCESS! Project pushed to GitHub!"
echo "============================================"
echo ""
echo "Your project is now live at:"
echo "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""
echo "Next steps:"
echo "  1. Go to your repo → Settings → Secrets → New secret"
echo "     Add: ANTHROPIC_API_KEY = (your key)"
echo "  2. Download ECHR dataset from Kaggle and place in data/"
echo "  3. Run: python -m src.retrieval.faiss_store --build"
echo "  4. Run: uvicorn src.api.main:app --reload"
echo ""
