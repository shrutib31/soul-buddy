"""
PostgreSQL connection pool configuration
This pool connects to the business data database (NOT the auth database)
Following principle of least privilege and database separation
"""

import os
import sys
from typing import Any, Dict, List, Optional, Callable
from contextlib import asynccontextmanager
import asyncpg
from dotenv import load_dotenv
import time
from urllib.parse import urlparse

load_dotenv()


def parse_postgres_url(url: str) -> Dict[str, str]:
    """
    Parse PostgreSQL connection URL
    
    Format: postgres://user:password@host:port/database
    
    Args:
        url: PostgreSQL connection URL
        
    Returns:
        Dictionary with host, port, database, user, password
    """
    parsed = urlparse(url)
    
    return {
        'host': parsed.hostname or 'localhost',
        'port': str(parsed.port or 5432),
        'database': parsed.path.lstrip('/') if parsed.path else 'postgres',
        'user': parsed.username or 'postgres',
        'password': parsed.password or ''
    }


class DatabaseConfig:
    """PostgreSQL database configuration and connection pool manager
    Supports both connection URL and individual environment variables
    """

    def __init__(self):
        """Initialize database configuration from connection URL or environment variables"""
        # Check for connection URL first (takes precedence)
        data_db_url = os.getenv('DATA_DB_URL', '')
        
        if data_db_url:
            # Parse connection URL
            config = parse_postgres_url(data_db_url)
            self.host = config['host']
            self.port = int(config['port'])
            self.database = config['database']
            self.user = config['user']
            self.password = config['password']
        else:
            # Fall back to individual environment variables
            self.host = os.getenv('DATA_DB_HOST', 'localhost')
            self.port = int(os.getenv('DATA_DB_PORT', '5432'))
            self.database = os.getenv('DATA_DB_NAME', 'fetch_api_db')
            self.user = os.getenv('DATA_DB_USER', 'fetch_user')
            self.password = os.getenv('DATA_DB_PASSWORD', '')
        
        self.max_connections = 20
        self.min_connections = 5
        self.command_timeout = 30.0
        self.pool: Optional[asyncpg.Pool] = None
        self.log_level = os.getenv('LOG_LEVEL', 'info')

    async def create_pool(self) -> asyncpg.Pool:
        """
        Create and configure database connection pool

        Returns:
            asyncpg.Pool: Configured connection pool
        """
        pool_config = {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'min_size': self.min_connections,
            'max_size': self.max_connections,
            'command_timeout': self.command_timeout,
        }

        # Only add password if it exists and is not empty (peer authentication if empty)
        if self.password and self.password.strip():
            pool_config['password'] = self.password

        try:
            self.pool = await asyncpg.create_pool(**pool_config)
            print('✅ Connected to PostgreSQL database')
            return self.pool
        except Exception as error:
            print(f'❌ Failed to create database pool: {error}')
            sys.exit(-1)

    async def close_pool(self):
        """Gracefully close all database connections"""
        if self.pool:
            try:
                await self.pool.close()
                print('✅ Database pool closed gracefully')
            except Exception as error:
                print(f'❌ Error closing database pool: {error}')
                raise

    async def query(self, text: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a query with automatic client management

        Args:
            text: SQL query text
            params: Query parameters (optional)

        Returns:
            List of row dictionaries

        Raises:
            Exception: If query execution fails
        """
        start = time.time()
        params = params or []

        try:
            if not self.pool:
                raise Exception('Database pool not initialized. Call create_pool() first.')

            async with self.pool.acquire() as connection:
                rows = await connection.fetch(text, *params)
                duration = (time.time() - start) * 1000  # Convert to milliseconds

                if self.log_level == 'debug':
                    print(f'Executed query: {text[:100]}... | Duration: {duration:.2f}ms | Rows: {len(rows)}')

                # Convert Record objects to dictionaries
                return [dict(row) for row in rows]

        except Exception as error:
            print(f'Database query error: {text[:100]}... | Error: {str(error)}')
            raise

    async def execute(self, text: str, params: Optional[List[Any]] = None) -> str:
        """
        Execute a command (INSERT, UPDATE, DELETE) with automatic client management

        Args:
            text: SQL command text
            params: Command parameters (optional)

        Returns:
            Status message from the database

        Raises:
            Exception: If command execution fails
        """
        start = time.time()
        params = params or []

        try:
            if not self.pool:
                raise Exception('Database pool not initialized. Call create_pool() first.')

            async with self.pool.acquire() as connection:
                result = await connection.execute(text, *params)
                duration = (time.time() - start) * 1000

                if self.log_level == 'debug':
                    print(f'Executed command: {text[:100]}... | Duration: {duration:.2f}ms | Result: {result}')

                return result

        except Exception as error:
            print(f'Database command error: {text[:100]}... | Error: {str(error)}')
            raise

    @asynccontextmanager
    async def get_client(self):
        """
        Get a client from the pool for transaction management
        Remember to release the client after use!

        Yields:
            Database connection

        Example:
            async with db.get_client() as client:
                await client.execute('SELECT 1')
        """
        if not self.pool:
            raise Exception('Database pool not initialized. Call create_pool() first.')

        connection = await self.pool.acquire()
        timeout_task = None

        try:
            # Set a timeout warning for long-held connections
            import asyncio

            async def warn_timeout():
                await asyncio.sleep(5)
                print('❌ Client has been checked out for more than 5 seconds!')

            timeout_task = asyncio.create_task(warn_timeout())
            yield connection
        finally:
            if timeout_task:
                timeout_task.cancel()
            await self.pool.release(connection)

    async def transaction(self, callback: Callable):
        """
        Execute a transaction with automatic rollback on error

        Args:
            callback: Async function to execute within transaction
                     The function receives the connection as an argument

        Returns:
            Result of the transaction callback

        Example:
            async def my_transaction(conn):
                await conn.execute('INSERT INTO ...')
                return await conn.fetch('SELECT ...')

            result = await db.transaction(my_transaction)
        """
        async with self.get_client() as client:
            async with client.transaction():
                try:
                    result = await callback(client)
                    return result
                except Exception as error:
                    # Transaction will auto-rollback on exception
                    raise

    async def test_connection(self) -> bool:
        """
        Test database connection

        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = await self.query(
                'SELECT NOW() as current_time, version() as postgres_version'
            )
            if result:
                row = result[0]
                version = row['postgres_version'].split(',')[0]
                print('✅ Database connection test successful')
                print(f'   PostgreSQL version: {version}')
                print(f'   Server time: {row["current_time"]}')
                return True
        except Exception as error:
            print(f'❌ Database connection test failed: {error}')
            return False

    async def initialize_schema(self):
        """
        Initialize database schema (create tables if they don't exist)
        Includes auth tables (roles, permissions, user_roles) and business data tables
        """
        try:
            print('🔧 Initializing database schema...')

            # ============================================
            # Business Data Tables Only
            # (Auth/RBAC tables are in separate auth database)
            # ============================================

            # Example: Create a sample resources table
            await self.execute('''
                CREATE TABLE IF NOT EXISTS resources (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    resource_type VARCHAR(50) NOT NULL,
                    data JSONB DEFAULT '{}',
                    metadata JSONB DEFAULT '{}',
                    created_by VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP NULL
                )
            ''')

            # Create index for better query performance
            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_resources_type
                ON resources(resource_type)
                WHERE deleted_at IS NULL
            ''')

            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_resources_created_by
                ON resources(created_by)
                WHERE deleted_at IS NULL
            ''')

            # Create a sample audit log table
            await self.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    resource_type VARCHAR(50),
                    resource_id INTEGER,
                    details JSONB DEFAULT '{}',
                    ip_address INET,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create index for audit logs
            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_logs_user_timestamp
                ON audit_logs(user_id, timestamp DESC)
            ''')

            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_logs_action
                ON audit_logs(action, timestamp DESC)
            ''')

            # Create intent_classifications table (Phase 3)
            await self.execute('''
                CREATE TABLE IF NOT EXISTS intent_classifications (
                    id SERIAL PRIMARY KEY,

                    -- User information
                    user_id VARCHAR(255) NOT NULL,
                    email VARCHAR(255),

                    -- Request data
                    message TEXT NOT NULL,
                    request_type VARCHAR(50) NOT NULL DEFAULT 'single',
                    request_metadata JSONB,

                    -- Classification results
                    intent VARCHAR(255) NOT NULL,
                    confidence DECIMAL(5, 4),
                    entities JSONB,
                    response_text TEXT,
                    reasoning JSONB,

                    -- Response metadata
                    processing_time_ms INTEGER,
                    model VARCHAR(100),
                    error TEXT,

                    -- Timestamps
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                    -- Constraints
                    CONSTRAINT intent_classifications_user_id_check CHECK (char_length(user_id) > 0),
                    CONSTRAINT intent_classifications_message_check CHECK (char_length(message) > 0),
                    CONSTRAINT intent_classifications_confidence_check CHECK (confidence >= 0 AND confidence <= 1)
                )
            ''')

            # Create indexes for intent_classifications
            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_intent_classifications_user_id
                ON intent_classifications(user_id)
            ''')

            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_intent_classifications_created_at
                ON intent_classifications(created_at DESC)
            ''')

            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_intent_classifications_intent
                ON intent_classifications(intent)
            ''')

            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_intent_classifications_user_created
                ON intent_classifications(user_id, created_at DESC)
            ''')

            # Create GIN indexes for JSONB columns
            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_intent_classifications_entities
                ON intent_classifications USING GIN (entities)
            ''')

            await self.execute('''
                CREATE INDEX IF NOT EXISTS idx_intent_classifications_reasoning
                ON intent_classifications USING GIN (reasoning)
            ''')

            print('✅ Database schema initialized successfully')
            return True
        except Exception as error:
            print(f'❌ Failed to initialize database schema: {error}')
            raise


# Global database instance
db_config = DatabaseConfig()


# Convenience functions for backward compatibility
async def query(text: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Execute a query with automatic client management"""
    return await db_config.query(text, params)


async def execute(text: str, params: Optional[List[Any]] = None) -> str:
    """Execute a command with automatic client management"""
    return await db_config.execute(text, params)


async def get_client():
    """Get a client from the pool for transaction management"""
    async with db_config.get_client() as client:
        yield client


async def transaction(callback: Callable):
    """Execute a transaction with automatic rollback on error"""
    return await db_config.transaction(callback)


async def test_connection() -> bool:
    """Test database connection"""
    return await db_config.test_connection()


async def initialize_schema():
    """Initialize database schema"""
    return await db_config.initialize_schema()


async def close_pool():
    """Gracefully close all database connections"""
    await db_config.close_pool()


# Export the database config instance and pool
__all__ = [
    'db_config',
    'query',
    'execute',
    'get_client',
    'transaction',
    'test_connection',
    'initialize_schema',
    'close_pool',
]
