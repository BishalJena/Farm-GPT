# Requirements & Dependencies Guide

## 📦 Overview

This project has three main dependency files:

1. **`/requirements.txt`** - Unified Python dependencies (backend)
2. **`/frontend/package.json`** - Frontend JavaScript dependencies
3. **`/backend/requirements.txt`** - Backend-specific Python dependencies (same as root)

---

## 🐍 Python Dependencies (Backend)

### Location
- **Root**: `/requirements.txt` (unified, recommended)
- **Backend**: `/backend/requirements.txt` (maintained for backward compatibility)

### Installation

```bash
# Option 1: Install from root (recommended)
pip install -r requirements.txt

# Option 2: Install from backend directory
cd backend
pip install -r requirements.txt
```

### Categories

#### Core Framework
- **fastapi** (0.110.1) - Web framework
- **uvicorn** (0.25.0) - ASGI server
- **starlette** (0.37.2) - ASGI toolkit

#### Database
- **motor** (3.3.1) - Async MongoDB driver
- **pymongo** (4.5.0) - MongoDB driver
- **redis** (5.0.1) - Redis cache

#### Authentication & Security
- **PyJWT** (2.10.1) - JWT tokens
- **bcrypt** (5.0.0) - Password hashing
- **python-jose** (3.5.0) - JOSE implementation
- **cryptography** (46.0.1) - Cryptographic recipes

#### AI & ML
- **deepgram-sdk** (3.7.0) - Speech-to-text (voice input)

#### HTTP & Async
- **httpx** (0.28.1) - Async HTTP client
- **aiofiles** (23.2.1) - Async file operations ⭐ **NEWLY ADDED**

#### Data Processing
- **pandas** (2.3.2) - Data analysis
- **numpy** (2.3.3) - Numerical computing
- **Pillow** (10.4.0) - Image processing

#### Validation
- **pydantic** (2.11.9) - Data validation
- **email-validator** (2.3.0) - Email validation

#### Caching
- **cachetools** (5.3.2) - Caching utilities
- **redis** (5.0.1) - Redis caching

#### Development Tools
- **black** (25.9.0) - Code formatter
- **flake8** (7.3.0) - Linter
- **mypy** (1.18.2) - Type checker
- **pytest** (8.4.2) - Testing framework
- **isort** (6.0.1) - Import sorting

---

## 📦 JavaScript Dependencies (Frontend)

### Location
`/frontend/package.json`

### Installation

```bash
cd frontend
npm install --legacy-peer-deps
```

### Key Dependencies

#### Core Framework
- **react** (19.0.0) - UI library
- **react-dom** (19.0.0) - React DOM renderer
- **react-scripts** (5.0.1) - Create React App scripts

#### UI Components (Radix UI)
- Complete set of accessible UI primitives
- 30+ components (@radix-ui/react-*)

#### Utilities
- **axios** (1.8.4) - HTTP client
- **react-router-dom** (7.5.1) - Routing
- **tailwindcss** (3.4.17) - Utility-first CSS
- **lucide-react** (0.507.0) - Icons
- **zod** (3.24.4) - Schema validation

#### Development
- **@craco/craco** (7.1.0) - Override CRA config
- **eslint** (9.23.0) - JavaScript linter
- **autoprefixer** (10.4.20) - CSS autoprefixer

---

## 🗂️ Root Dependencies

### Location
`/package.json` (root level)

### Purpose
Development tooling only (currently just shadcn UI CLI)

### Installation

```bash
npm install
```

### Contents
```json
{
  "devDependencies": {
    "shadcn": "^3.3.1"
  }
}
```

**Note**: This is separate from frontend/package.json and is only for development tooling.

---

## 🔍 Dependency Audit Results

### ✅ All Required Packages Verified

#### Backend Python Packages
All imports in the codebase are covered:
- ✓ fastapi, motor, pymongo, redis
- ✓ PyJWT, bcrypt, passlib, python-jose
- ✓ httpx, httpcore, h11
- ✓ pandas, numpy, Pillow
- ✓ pydantic, email-validator
- ✓ python-dotenv, boto3, requests
- ✓ deepgram-sdk (voice input)
- ✓ cachetools (performance)
- ✓ aiofiles ⭐ **NEWLY ADDED** (was missing)

#### Frontend JavaScript Packages
All required for the UI:
- ✓ React 19 and ecosystem
- ✓ Radix UI components (complete set)
- ✓ Tailwind CSS and utilities
- ✓ Axios for API calls
- ✓ React Router for navigation

---

## 🆕 Recently Added Dependencies

### Backend
1. **deepgram-sdk** (3.7.0)
   - Purpose: Speech-to-text voice input
   - Added: Voice feature implementation
   - Required by: `voice_stt_service.py`

2. **aiofiles** (23.2.1) ⭐ **FIXED**
   - Purpose: Async file operations
   - Was imported but missing from requirements
   - Required by: `server.py` (media upload handling)

### Frontend
No new dependencies in recent changes (Auth UI used existing CSS)

---

## 📋 Installation Checklist

### Fresh Install

```bash
# 1. Clone repository
git clone <repo-url>
cd emergent-farmchat

# 2. Install Python dependencies (backend)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Install Node dependencies (frontend)
cd frontend
npm install --legacy-peer-deps
cd ..

# 4. Install root dev tools (optional)
npm install

# 5. Setup environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# Edit .env files with your API keys

# 6. Verify installations
python backend/test_imports.py
cd frontend && npm test -- --watchAll=false
```

---

## 🔧 Maintenance

### Update All Dependencies

```bash
# Backend Python
pip list --outdated
pip install --upgrade -r requirements.txt

# Frontend JavaScript
cd frontend
npm outdated
npm update
```

### Add New Dependencies

#### Backend Python
```bash
# Install package
pip install package-name

# Add to requirements.txt
pip freeze | grep package-name >> requirements.txt

# Or manually add with version
echo "package-name==1.0.0" >> requirements.txt
```

#### Frontend JavaScript
```bash
cd frontend
npm install --save package-name
# Automatically updates package.json
```

---

## ⚠️ Known Issues

### Frontend Peer Dependencies
Use `--legacy-peer-deps` flag due to React 19:
```bash
npm install --legacy-peer-deps
```

### Python Version
Requires Python 3.8+. Recommended: Python 3.11 or 3.12

### Node Version
Requires Node.js 16+. Recommended: Node.js 18 or 20

---

## 📊 Dependency Statistics

### Backend (Python)
- **Total packages**: 77
- **Production**: ~50
- **Development**: ~27
- **Installation size**: ~500-800 MB

### Frontend (JavaScript)
- **Total packages**: ~2000 (including nested dependencies)
- **Direct dependencies**: 37
- **Dev dependencies**: 6
- **Installation size**: ~400-600 MB

---

## 🚀 Production Deployment

### Backend Requirements
For production, consider splitting into:
- `requirements.txt` - All dependencies
- `requirements-prod.txt` - Production only (exclude dev tools)
- `requirements-dev.txt` - Development tools only

### Example Split

**requirements-prod.txt**:
```txt
fastapi==0.110.1
uvicorn==0.25.0
motor==3.3.1
# ... (all production packages)
```

**requirements-dev.txt**:
```txt
-r requirements-prod.txt
black==25.9.0
flake8==7.3.0
pytest==8.4.2
mypy==1.18.2
```

### Frontend Production Build
```bash
cd frontend
npm run build
# Creates optimized production build in /frontend/build/
```

---

## 🔗 Related Files

- `/backend/requirements.txt` - Backend dependencies
- `/frontend/package.json` - Frontend dependencies
- `/package.json` - Root dev dependencies
- `/backend/.env` - Backend configuration
- `/frontend/.env` - Frontend configuration
- `/.gitignore` - Ignored files and dependencies

---

## 📞 Support

### Missing Dependencies?
If you see import errors:
1. Check this guide for the package
2. Install: `pip install <package-name>`
3. Update requirements.txt
4. Report the issue

### Version Conflicts?
1. Check Python version: `python --version`
2. Check Node version: `node --version`
3. Use virtual environment for Python
4. Clear node_modules and reinstall

---

**Last Updated**: October 5, 2025
**Maintained By**: Development Team
**Status**: ✅ All dependencies verified and documented
