import sys
from pathlib import Path

# Add parent directory to path so we can import config and orm
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging, get_logger
from config.sqlalchemy_db import data_db_sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# import all models to register them with Base
from orm.base import Base
from orm import *


def main():
    # Initialize logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("🗑️  CLEANING UP DATABASE")
    logger.info("=" * 80)
    
    try:
        # Step 1: Initialize data_db_sqlalchemy connection
        logger.info("Step 1: Initializing database connection...")
        if data_db_sqlalchemy is None:
            from config.sqlalchemy_db import SQLAlchemyDataDB
            data_db = SQLAlchemyDataDB()
            db_url = data_db.database_url
            logger.info(f"Database URL: {db_url.split('@')[1] if '@' in db_url else db_url}")
        else:
            db_url = data_db_sqlalchemy.database_url
        
        # Convert asyncpg URL to sync psycopg2 for synchronous operations
        sync_db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
        logger.info(f"✅ Using sync URL: {sync_db_url.split('@')[1] if '@' in sync_db_url else sync_db_url}")
        
        # Create sync engine
        engine = create_engine(sync_db_url, echo=False)
        
        # Step 2: Drop all tables
        logger.info("Step 2: Dropping all database tables...")
        logger.warning("⚠️  This will delete ALL data in the database!")
        
        # Drop all tables in reverse order of dependencies
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ All tables dropped successfully")
        
        # Close engine
        engine.dispose()
        
        logger.info("=" * 80)
        logger.info("✅ DATABASE CLEANUP COMPLETE")
        logger.info("=" * 80)
    except Exception as error:
        logger.error(f"❌ Database cleanup failed: {error}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
