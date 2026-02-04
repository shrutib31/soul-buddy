"""
PostgreSQL connection pool configuration for Authentication/RBAC database
This pool connects to the authentication database (separate from business data database)
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


class AuthDatabaseConfig:
    """PostgreSQL database configuration and connection pool manager for Auth/RBAC database
    Supports both connection URL and individual environment variables
    """

    def __init__(self):
        """Initialize database configuration from connection URL or environment variables"""
        # Check for connection URL first (takes precedence)
        auth_db_url = os.getenv('AUTH_DB_URL', os.getenv('RBAC_DB_URL', ''))
        
        if auth_db_url:
            # Parse connection URL
            config = parse_postgres_url(auth_db_url)
            self.host = config['host']
            self.port = int(config['port'])
            self.database = config['database']
            self.user = config['user']
            self.password = config['password']
        else:
            # Fall back to individual environment variables
            self.host = os.getenv('AUTH_DB_HOST', os.getenv('RBAC_DB_HOST', 'localhost'))
            self.port = int(os.getenv('AUTH_DB_PORT', os.getenv('RBAC_DB_PORT', '5432')))
            self.database = os.getenv('AUTH_DB_NAME', os.getenv('RBAC_DB_NAME', 'auth_db'))
            self.user = os.getenv('AUTH_DB_USER', os.getenv('RBAC_DB_USER', 'postgres'))
            self.password = os.getenv('AUTH_DB_PASSWORD', os.getenv('RBAC_DB_PASSWORD', ''))
        
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
            print(f'✅ Connected to Auth/RBAC PostgreSQL database: {self.database}')
            return self.pool
        except Exception as error:
            print(f'❌ Failed to create auth database pool: {error}')
            sys.exit(-1)

    async def close_pool(self):
        """Gracefully close all database connections"""
        if self.pool:
            try:
                await self.pool.close()
                print('✅ Auth database pool closed gracefully')
            except Exception as error:
                print(f'❌ Error closing auth database pool: {error}')
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
                raise Exception('Auth database pool not initialized. Call create_pool() first.')

            async with self.pool.acquire() as connection:
                rows = await connection.fetch(text, *params)
                duration = (time.time() - start) * 1000  # Convert to milliseconds

                if self.log_level == 'debug':
                    print(f'[Auth DB] Executed query: {text[:100]}... | Duration: {duration:.2f}ms | Rows: {len(rows)}')

                # Convert Record objects to dictionaries
                return [dict(row) for row in rows]

        except Exception as error:
            print(f'[Auth DB] Database query error: {text[:100]}... | Error: {str(error)}')
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
                raise Exception('Auth database pool not initialized. Call create_pool() first.')

            async with self.pool.acquire() as connection:
                result = await connection.execute(text, *params)
                duration = (time.time() - start) * 1000

                if self.log_level == 'debug':
                    print(f'[Auth DB] Executed command: {text[:100]}... | Duration: {duration:.2f}ms | Result: {result}')

                return result

        except Exception as error:
            print(f'[Auth DB] Database command error: {text[:100]}... | Error: {str(error)}')
            raise

    @asynccontextmanager
    async def get_client(self):
        """
        Get a client from the pool for transaction management

        Yields:
            Database connection
        """
        if not self.pool:
            raise Exception('Auth database pool not initialized. Call create_pool() first.')

        connection = await self.pool.acquire()
        timeout_task = None

        try:
            import asyncio

            async def warn_timeout():
                await asyncio.sleep(5)
                print('❌ Auth DB client has been checked out for more than 5 seconds!')

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
                print('✅ Auth database connection test successful')
                print(f'   PostgreSQL version: {version}')
                print(f'   Server time: {row["current_time"]}')
                return True
        except Exception as error:
            print(f'❌ Auth database connection test failed: {error}')
            return False

# Global instance
auth_db_config = AuthDatabaseConfig()

