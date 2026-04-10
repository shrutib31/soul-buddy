#!/usr/bin/env bash
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[soul-buddy]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[soul-buddy]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[soul-buddy]${RESET} $*"; }
error()   { echo -e "${RED}${BOLD}[soul-buddy]${RESET} $*"; exit 1; }

# ── 1. Resolve script directory ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}  💙 SoulBuddy Frontend${RESET}"
echo "  ─────────────────────────────"
echo ""

# ── 2. Check Node ─────────────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
  error "Node.js is not installed. Install it from https://nodejs.org (v18+ recommended)."
fi

NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
  warn "Node.js v$NODE_VERSION detected. v18+ is recommended."
fi

info "Node $(node -v)  |  npm $(npm -v)"

# ── 3. Env file ───────────────────────────────────────────────────────────────
if [ ! -f ".env.local" ]; then
  if [ -f ".env.example" ]; then
    warn ".env.local not found — copying from .env.example"
    cp .env.example .env.local
    warn "Open ${BOLD}sb-frontend/.env.local${RESET}${YELLOW} and fill in your Supabase credentials before the app will work."
  else
    warn ".env.local not found and no .env.example to copy from. Create sb-frontend/.env.local with:"
    echo ""
    echo "  VITE_SUPABASE_URL=https://your-project.supabase.co"
    echo "  VITE_SUPABASE_ANON_KEY=your-anon-key"
    echo "  VITE_API_BASE_URL=http://localhost:8000"
    echo ""
  fi
else
  success ".env.local found"
fi

# ── 4. Install dependencies ───────────────────────────────────────────────────
if [ ! -d "node_modules" ]; then
  info "node_modules not found — installing dependencies..."
  npm install
  success "Dependencies installed"
else
  info "node_modules present — skipping install (run 'npm install' manually to update)"
fi

# ── 5. Start dev server ───────────────────────────────────────────────────────
echo ""
success "Starting dev server..."
echo ""
npm run dev
