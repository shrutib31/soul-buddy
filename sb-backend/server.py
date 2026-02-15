"""
Fast API server starting script

This module handles the initialization and startup of the SoulBuddy backend server.
It performs the following tasks before starting the FastAPI application:

1. Logging Configuration
   - Initializes logging system with both console and file handlers
   - Supports timestamped log files for better organization
   - Logs to both business data and audit logs

2. Database Configuration
   - Initializes connection pool for the primary data database
   - Initializes connection pool for the authentication/RBAC database
   - Tests both database connections to ensure connectivity

3. Supabase Configuration
   - Initializes Supabase client for real-time and authentication services
   - Tests connection to verify API credentials

4. Startup & Shutdown Events
   - Sets up async context managers for graceful startup/shutdown
   - Ensures all resources are properly released on server shutdown

Note: All configurations must be successful before the server starts. If any
configuration or connection test fails, the server will exit with status code -1.
"""

import asyncio
import sys
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import configuration modules
from config.logging_config import setup_logging, get_logger
from config.database import DatabaseConfig
from config.auth_database import AuthDatabaseConfig
from config.sqlalchemy_db import SQLAlchemyDataDB, SQLAlchemyAuthDB
from config.supabase import test_connection as test_supabase_connection

# Load environment variables
load_dotenv()

# Global instances for database connections
data_db: Optional[DatabaseConfig] = None
auth_db: Optional[AuthDatabaseConfig] = None
data_db_sqlalchemy: Optional[SQLAlchemyDataDB] = None
auth_db_sqlalchemy: Optional[SQLAlchemyAuthDB] = None

# Initialize logger
logger = get_logger(__name__)


# ============================================================================
# STARTUP & SHUTDOWN EVENT HANDLERS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI application.
    
    Startup: Initializes all configurations and tests connections
    Shutdown: Gracefully closes all database connections
    
    Args:
        app: FastAPI application instance
    """
    # ========================================================================
    # STARTUP PHASE: Initialize all configurations
    # ========================================================================
    
    startup_success = await initialize_all_configurations()
    
    if not startup_success:
        logger.critical("❌ Failed to initialize one or more configurations. Server startup aborted.")
        sys.exit(-1)
    
    logger.info("✅ Server startup phase completed successfully. All configurations initialized.")
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 SERVER READY")
    logger.info("=" * 80)
    logger.info("📍 Server accessible at:")
    logger.info("   • Localhost: http://localhost:8000")
    logger.info("   • Local IP: http://127.0.0.1:8000")
    logger.info("   • Network: http://0.0.0.0:8000")
    logger.info("")
    logger.info("📖 API Documentation:")
    logger.info("   • Interactive Swagger UI: http://localhost:8000/docs")
    logger.info("   • Alternative ReDoc UI: http://localhost:8000/redoc")
    logger.info("   • OpenAPI JSON Schema: http://localhost:8000/openapi.json")
    logger.info("")
    logger.info("🏥 Health Check: http://localhost:8000/health")
    logger.info("")
    logger.info("💬 Chat Endpoints:")
    logger.info("   • POST /api/v1/chat/incognito - Anonymous chat")
    logger.info("   • POST /api/v1/chat/incognito/stream - Anonymous streaming chat")
    logger.info("   • POST /api/v1/chat/cognito - Authenticated chat")
    logger.info("   • POST /api/v1/chat/cognito/stream - Authenticated streaming chat")
    logger.info("=" * 80)
    logger.info("")
    
    yield  # Application runs here
    
    # ========================================================================
    # SHUTDOWN PHASE: Clean up resources
    # ========================================================================
    
    logger.info("🛑 Server shutdown initiated. Cleaning up resources...")
    await cleanup_all_resources()
    logger.info("✅ All resources cleaned up. Server shutdown complete.")


# ============================================================================
# CONFIGURATION INITIALIZATION FUNCTIONS
# ============================================================================

async def initialize_logging() -> bool:
    """
    Initialize logging configuration.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("\n[1/5] Initializing Logging Configuration...")
        setup_logging()
        logger.info("✅ Logging configuration initialized successfully")
        return True
    except Exception as error:
        logger.critical(f"❌ Failed to initialize logging configuration: {error}")
        return False


async def initialize_data_database() -> bool:
    """
    Initialize the primary data database (business data).
    Creates connection pool and tests connectivity.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("\n[2/5] Initializing Data Database Configuration...")
        global data_db
        data_db = DatabaseConfig()
        
        # Log database connection details (without password for security)
        logger.info(f"   Database Host: {data_db.host}:{data_db.port}")
        logger.info(f"   Database Name: {data_db.database}")
        logger.info(f"   Database User: {data_db.user}")
        logger.info(f"   Connection Pool: min={data_db.min_connections}, max={data_db.max_connections}")
        
        # Create connection pool
        logger.info("   Creating connection pool...")
        await data_db.create_pool()
        
        # Test connection
        logger.info("   Testing database connection...")
        connection_test_passed = await data_db.test_connection()
        
        if not connection_test_passed:
            logger.error("❌ Data database connection test failed")
            return False
        
        logger.info("✅ Data database initialized and verified successfully")
        logger.info(f"   Pool Status: {data_db.pool._holders.__len__()} connections available")
        return True
        
    except Exception as error:
        logger.critical(f"❌ Failed to initialize data database: {error}")
        return False


async def initialize_auth_database() -> bool:
    """
    Initialize the authentication/RBAC database.
    Creates connection pool and tests connectivity.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("\n[3/5] Initializing Authentication/RBAC Database Configuration...")
        global auth_db
        auth_db = AuthDatabaseConfig()
        
        # Log database connection details (without password for security)
        logger.info(f"   Database Host: {auth_db.host}:{auth_db.port}")
        logger.info(f"   Database Name: {auth_db.database}")
        logger.info(f"   Database User: {auth_db.user}")
        logger.info(f"   Connection Pool: min={auth_db.min_connections}, max={auth_db.max_connections}")
        
        # Create connection pool
        logger.info("   Creating connection pool...")
        await auth_db.create_pool()
        
        # Test connection
        logger.info("   Testing database connection...")
        connection_test_passed = await auth_db.test_connection()
        
        if not connection_test_passed:
            logger.error("❌ Auth database connection test failed")
            return False
        
        logger.info("✅ Auth database initialized and verified successfully")
        logger.info(f"   Pool Status: {auth_db.pool._holders.__len__()} connections available")
        return True
        
    except Exception as error:
        logger.critical(f"❌ Failed to initialize auth database: {error}")
        return False


async def initialize_sqlalchemy_engines() -> bool:
    """
    Initialize SQLAlchemy engines for both databases.
    Provides ORM capabilities for database transactions.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("\n[4/5] Initializing SQLAlchemy Engines...")
        global data_db_sqlalchemy, auth_db_sqlalchemy
        
        # Initialize Data DB SQLAlchemy
        logger.info("   Initializing Data Database SQLAlchemy engine...")
        data_db_sqlalchemy = SQLAlchemyDataDB()
        await data_db_sqlalchemy.init_engine()
        logger.info("✅ Data Database SQLAlchemy engine initialized")
        
        # Initialize Auth DB SQLAlchemy
        logger.info("   Initializing Auth Database SQLAlchemy engine...")
        auth_db_sqlalchemy = SQLAlchemyAuthDB()
        await auth_db_sqlalchemy.init_engine()
        logger.info("✅ Auth Database SQLAlchemy engine initialized")
        
        logger.info("✅ All SQLAlchemy engines initialized successfully")
        return True
        
    except Exception as error:
        logger.critical(f"❌ Failed to initialize SQLAlchemy engines: {error}")
        return False


async def initialize_supabase() -> bool:
    """
    Initialize Supabase client and test connection.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("\n[5/5] Initializing Supabase Configuration...")
        logger.info("   Testing Supabase connection...")
        
        supabase_test_passed = await test_supabase_connection()
        
        if not supabase_test_passed:
            logger.error("❌ Supabase connection test failed")
            return False
        
        logger.info("✅ Supabase initialized and verified successfully")
        return True
        
    except Exception as error:
        logger.critical(f"❌ Failed to initialize Supabase: {error}")
        return False


async def initialize_all_configurations() -> bool:
    """
    Initialize all configurations in the correct order:
    1. Logging configuration
    2. Data database configuration and connection testing
    3. Authentication database configuration and connection testing
    4. SQLAlchemy engines for both databases
    5. Supabase configuration and connection testing
    
    Returns:
        True if all configurations initialized successfully, False otherwise
    """
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING SERVER INITIALIZATION SEQUENCE")
    logger.info("=" * 80)
    
    # Execute initialization steps in sequence
    initialization_steps = [
        ("Logging", initialize_logging),
        ("Data Database", initialize_data_database),
        ("Auth Database", initialize_auth_database),
        ("SQLAlchemy Engines", initialize_sqlalchemy_engines),
        ("Supabase", initialize_supabase),
    ]
    
    for step_name, step_function in initialization_steps:
        success = await step_function()
        if not success:
            logger.critical(f"❌ {step_name} initialization failed. Aborting startup.")
            return False
    
    # ========================================================================
    # All configurations initialized successfully
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL CONFIGURATIONS INITIALIZED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\n******* SYSTEM STATUS:*******")
    logger.info(f"   ✅ Logging System: Active (with timestamped logs)")
    logger.info(f"   ✅ Data Database Pool: Ready ({data_db.pool._holders.__len__()} connections)")
    logger.info(f"   ✅ Auth Database Pool: Ready ({auth_db.pool._holders.__len__()} connections)")
    logger.info(f"   ✅ Data Database SQLAlchemy: Engine initialized")
    logger.info(f"   ✅ Auth Database SQLAlchemy: Engine initialized")
    logger.info(f"   ✅ Supabase Client: Connected")
    logger.info("\n Ready to serve requests!\n")
    
    return True


async def cleanup_all_resources() -> None:
    """
    Clean up all resources during server shutdown.
    
    Closes:
    1. Data database connection pool
    2. Authentication database connection pool
    3. SQLAlchemy engines
    4. Supabase client connections (implicit)
    """
    
    logger.info("🧹 Cleaning up resources...")
    
    # Close data database pool
    if data_db is not None:
        try:
            logger.info("   Closing data database connection pool...")
            await data_db.close_pool()
        except Exception as error:
            logger.error(f"   Error closing data database pool: {error}")
    
    # Close auth database pool
    if auth_db is not None:
        try:
            logger.info("   Closing auth database connection pool...")
            await auth_db.close_pool()
        except Exception as error:
            logger.error(f"   Error closing auth database pool: {error}")
    
    # Close SQLAlchemy engines
    if data_db_sqlalchemy is not None:
        try:
            logger.info("   Closing Data Database SQLAlchemy engine...")
            await data_db_sqlalchemy.close_engine()
        except Exception as error:
            logger.error(f"   Error closing Data DB SQLAlchemy engine: {error}")
    
    if auth_db_sqlalchemy is not None:
        try:
            logger.info("   Closing Auth Database SQLAlchemy engine...")
            await auth_db_sqlalchemy.close_engine()
        except Exception as error:
            logger.error(f"   Error closing Auth DB SQLAlchemy engine: {error}")
    
    logger.info("✅ Resource cleanup complete")


# ============================================================================
# FastAPI Application Setup
# ============================================================================

# Create FastAPI application with lifespan management
app = FastAPI(
    title="SoulBuddy Backend API",
    description="Backend API for the SoulBuddy application",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Router Registration
# ============================================================================

# Import and register API routers
from api.chat import router as chat_router
from api.classify import router as classify_router

app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.include_router(classify_router, prefix="/api/v1", tags=["Classification"])


# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server and database connectivity.
    
    Returns:
        Status of the server and all connected services
    """
    return {
        "status": "healthy",
        "server": "running",
        "data_database": "connected" if data_db and data_db.pool else "disconnected",
        "auth_database": "connected" if auth_db and auth_db.pool else "disconnected",
        "supabase": "connected"
    }


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None  # Use our custom logging configuration
    )
