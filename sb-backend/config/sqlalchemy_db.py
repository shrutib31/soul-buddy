"""
SQLAlchemy Database Configuration and Session Management

This module provides SQLAlchemy-based database access with transaction support.
It works alongside the asyncpg connection pools for flexibility in database operations.

Features:
- Async SQLAlchemy engine and session management
- Context managers for automatic transaction handling
- Support for both data and auth databases
- Connection pooling with configurable parameters
"""

import os
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# SQLAlchemy Base Models
# ============================================================================

# Base class for data database models
DataBase = declarative_base()

# Base class for auth database models
AuthBase = declarative_base()


# ============================================================================
# Database Engine Configuration
# ============================================================================

class SQLAlchemyDataDB:
    """SQLAlchemy configuration for the primary data database"""
    
    def __init__(self):
        """Initialize SQLAlchemy engine from environment variables"""
        # Get database URL from environment
        db_url = os.getenv('DATA_DB_URL', '')
        
        if not db_url:
            # Construct URL from individual parameters
            host = os.getenv('DATA_DB_HOST', 'localhost')
            port = os.getenv('DATA_DB_PORT', '5432')
            database = os.getenv('DATA_DB_NAME', 'soulbuddy_data_db')
            user = os.getenv('DATA_DB_USER', 'postgres')
            password = os.getenv('DATA_DB_PASSWORD', '')
            
            # Use asyncpg driver
            if password:
                db_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
            else:
                db_url = f"postgresql+asyncpg://{user}@{host}:{port}/{database}"
        else:
            # Convert postgres:// to postgresql+asyncpg://
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql+asyncpg://', 1)
            elif db_url.startswith('postgresql://'):
                db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        
        self.database_url = db_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self.log_level = os.getenv('LOG_LEVEL', 'info')
    
    async def init_engine(self) -> AsyncEngine:
        """
        Initialize the SQLAlchemy async engine with connection pooling
        
        Returns:
            AsyncEngine: Configured async engine
        """
        if self.engine is None:
            self.engine = create_async_engine(
                self.database_url,
                echo=(self.log_level == 'debug'),  # SQL query logging
                pool_size=20,  # Maximum number of connections
                max_overflow=10,  # Additional connections beyond pool_size
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            if self.log_level == 'debug':
                print(f"✅ SQLAlchemy Data DB engine initialized: {self.database_url.split('@')[1]}")
        
        return self.engine
    
    async def close_engine(self):
        """Gracefully close the engine and all connections"""
        if self.engine:
            await self.engine.dispose()
            if self.log_level == 'debug':
                print("✅ SQLAlchemy Data DB engine closed")
            self.engine = None
            self.session_factory = None
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic transaction management
        
        Usage:
            async with data_db_sqlalchemy.get_session() as session:
                result = await session.execute(query)
                await session.commit()  # Explicit commit
        
        Yields:
            AsyncSession: Database session
        """
        if self.session_factory is None:
            await self.init_engine()
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic commit/rollback
        
        Usage:
            async with data_db_sqlalchemy.get_transaction() as session:
                session.add(new_object)
                # Automatic commit on success, rollback on exception
        
        Yields:
            AsyncSession: Database session with automatic transaction handling
        """
        if self.session_factory is None:
            await self.init_engine()
        
        async with self.session_factory() as session:
            async with session.begin():
                try:
                    yield session
                except Exception:
                    await session.rollback()
                    raise


class SQLAlchemyAuthDB:
    """SQLAlchemy configuration for the authentication/RBAC database"""
    
    def __init__(self):
        """Initialize SQLAlchemy engine from environment variables"""
        # Get database URL from environment
        db_url = os.getenv('AUTH_DB_URL', os.getenv('RBAC_DB_URL', ''))
        
        if not db_url:
            # Construct URL from individual parameters
            host = os.getenv('AUTH_DB_HOST', os.getenv('RBAC_DB_HOST', 'localhost'))
            port = os.getenv('AUTH_DB_PORT', os.getenv('RBAC_DB_PORT', '5432'))
            database = os.getenv('AUTH_DB_NAME', os.getenv('RBAC_DB_NAME', 'souloxy-db'))
            user = os.getenv('AUTH_DB_USER', os.getenv('RBAC_DB_USER', 'postgres'))
            password = os.getenv('AUTH_DB_PASSWORD', os.getenv('RBAC_DB_PASSWORD', ''))
            
            # Use asyncpg driver
            if password:
                db_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
            else:
                db_url = f"postgresql+asyncpg://{user}@{host}:{port}/{database}"
        else:
            # Convert postgres:// to postgresql+asyncpg://
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql+asyncpg://', 1)
            elif db_url.startswith('postgresql://'):
                db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        
        self.database_url = db_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self.log_level = os.getenv('LOG_LEVEL', 'info')
    
    async def init_engine(self) -> AsyncEngine:
        """
        Initialize the SQLAlchemy async engine with connection pooling
        
        Returns:
            AsyncEngine: Configured async engine
        """
        if self.engine is None:
            self.engine = create_async_engine(
                self.database_url,
                echo=(self.log_level == 'debug'),  # SQL query logging
                pool_size=20,  # Maximum number of connections
                max_overflow=10,  # Additional connections beyond pool_size
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            if self.log_level == 'debug':
                print(f"✅ SQLAlchemy Auth DB engine initialized: {self.database_url.split('@')[1]}")
        
        return self.engine
    
    async def close_engine(self):
        """Gracefully close the engine and all connections"""
        if self.engine:
            await self.engine.dispose()
            if self.log_level == 'debug':
                print("✅ SQLAlchemy Auth DB engine closed")
            self.engine = None
            self.session_factory = None
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic transaction management
        
        Usage:
            async with auth_db_sqlalchemy.get_session() as session:
                result = await session.execute(query)
                await session.commit()  # Explicit commit
        
        Yields:
            AsyncSession: Database session
        """
        if self.session_factory is None:
            await self.init_engine()
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic commit/rollback
        
        Usage:
            async with auth_db_sqlalchemy.get_transaction() as session:
                session.add(new_object)
                # Automatic commit on success, rollback on exception
        
        Yields:
            AsyncSession: Database session with automatic transaction handling
        """
        if self.session_factory is None:
            await self.init_engine()
        
        async with self.session_factory() as session:
            async with session.begin():
                try:
                    yield session
                except Exception:
                    await session.rollback()
                    raise


# ============================================================================
# Global Instances
# ============================================================================

# Global SQLAlchemy instances (initialized in server.py)
data_db_sqlalchemy: Optional[SQLAlchemyDataDB] = None
auth_db_sqlalchemy: Optional[SQLAlchemyAuthDB] = None


# ============================================================================
# Utility Functions
# ============================================================================

async def init_all_engines():
    """Initialize all SQLAlchemy engines"""
    global data_db_sqlalchemy, auth_db_sqlalchemy
    
    data_db_sqlalchemy = SQLAlchemyDataDB()
    await data_db_sqlalchemy.init_engine()
    
    auth_db_sqlalchemy = SQLAlchemyAuthDB()
    await auth_db_sqlalchemy.init_engine()
    
    print("✅ All SQLAlchemy engines initialized")


async def close_all_engines():
    """Close all SQLAlchemy engines"""
    if data_db_sqlalchemy:
        await data_db_sqlalchemy.close_engine()
    
    if auth_db_sqlalchemy:
        await auth_db_sqlalchemy.close_engine()
    
    print("✅ All SQLAlchemy engines closed")
