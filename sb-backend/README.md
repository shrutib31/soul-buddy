# SoulBuddy Backend

Backend API server for the SoulBuddy application.

## 1) How to Initialize the DB

1. Create the PostgreSQL databases:

```bash
psql -U postgres
CREATE DATABASE soulbuddy;
CREATE DATABASE "souloxy-db";
\q
```

2. Initialize schema and seed config data:

```bash
python scripts/init_db.py
```

3. (Optional) Drop all tables:

```bash
python scripts/cleanup_db.py
```

## 2) Where to Keep the APIs

All API endpoints live in the `api/` directory.

```
sb-backend/
├── api/
│   ├── __init__.py
│   ├── chat.py
│   └── <your_api_module>.py
```

Register new routers in `server.py`:

```python
from api.example import router as example_router

app.include_router(example_router, prefix="/api/v1", tags=["Example"])
```

## 3) Where the Code Hits from the API

High-level flow of a request:

```
HTTP Request
  → server.py (FastAPI app + router registration)
  → api/<module>.py (endpoint handler + Pydantic validation)
  → Database access:
       - ORM: orm/models.py + config/sqlalchemy_db.py
       - SQL: config/database.py
  → HTTP Response
```

Quick reference:

- API handlers: `api/<module>.py`
- Router registration: `server.py`
- ORM models: `orm/models.py`
- SQLAlchemy sessions: `config/sqlalchemy_db.py`
- Direct SQL pool: `config/database.py`

The setup script will automatically detect if UV is installed and use it for faster dependency management:

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup with UV (Faster)

If you have UV installed:

```bash
# Create virtual environment
uv venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# Create logs directory
mkdir -p logs
```

### Option 3: Manual Setup with venv (Traditional)

If you don't have UV:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create logs directory
mkdir -p logs
```

## Installing UV (Optional but Recommended)

UV is a fast Python package installer and resolver, written in Rust. It's much faster than pip:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using Homebrew (macOS)
brew install uv
```

## Configuration

1. Copy and configure the `.env` file with your database and Supabase credentials:

```bash
# The .env file should already exist with all required variables
# Update the values according to your environment:
# - Database URLs (DATA_DB_URL, AUTH_DB_URL)
# - Supabase credentials (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_ANON_KEY)
# - Logging configuration (LOG_LEVEL, LOG_DIR)
```

2. Ensure your PostgreSQL databases are running and accessible

## Running the Server

### Development Mode

```bash
# Activate virtual environment first
source venv/bin/activate

# Option 1: Direct Python
python server.py

# Option 2: Uvicorn with auto-reload
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

## Project Structure

```
sb-backend/
├── server.py              # Main FastAPI application
├── config/                # Configuration modules
│   ├── database.py        # Data database configuration
│   ├── auth_database.py   # Auth database configuration
│   ├── supabase.py        # Supabase configuration
│   └── logging.config.py  # Logging configuration
├── config-files/          # Configuration files
│   └── logging.yaml       # Logging YAML configuration
├── logs/                  # Log files (auto-generated)
├── .env                   # Environment variables
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── setup.sh              # Setup script

```

## API Endpoints

### Health Check

```bash
GET /health

Response:
{
    "status": "healthy",
    "server": "running",
    "data_database": "connected",
    "auth_database": "connected",
    "supabase": "connected"
}
```

## Development

### Code Quality Tools

The project includes several development tools:

```bash
# Format code with Black
black .

# Lint with Ruff
ruff check .

# Type check with MyPy
mypy .

# Run tests
pytest
```

## Environment Variables

See `.env` file for all available environment variables. Key variables include:

- `DATA_DB_URL` - Primary data database connection URL
- `AUTH_DB_URL` - Authentication database connection URL
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key
- `SUPABASE_ANON_KEY` - Supabase anonymous key
- `LOG_LEVEL` - Logging level (debug/info/warning/error/critical)
- `LOG_DIR` - Directory for log files

## Logging

The application uses a comprehensive logging system:

- **Console Logging** - Real-time logs in the terminal
- **File Logging** - Timestamped log files in the `logs/` directory
- **Audit Logging** - Separate audit trail for security events
- **Log Rotation** - Automatic daily log rotation with configurable retention

Log files are named with timestamps (e.g., `soulbuddy_2026-02-03_14-30-45.log`) for easy tracking.

## Database Setup

The server automatically:
1. Initializes connection pools for both databases
2. Tests database connectivity
3. Verifies Supabase connection
4. Exits gracefully if any connection fails

Ensure your databases are created and accessible before starting the server.

## License

MIT
