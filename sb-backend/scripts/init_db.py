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
from seed.seed_config import seed_all_config


def main():
    # Initialize logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("🔧 INITIALIZING DATABASE")
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
        SessionLocal = sessionmaker(bind=engine)
        
        # Step 2: Create schema
        logger.info("Step 2: Creating database schema...")
        
        # First, enable uuid-ossp extension
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
            conn.commit()
            logger.info("✅ PostgreSQL uuid-ossp extension enabled")
        
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Schema created successfully")
        
        # Step 3: Seed configuration data
        logger.info("Step 3: Seeding configuration data...")
        db = SessionLocal()
        seed_all_config(db)
        db.close()
        logger.info("✅ Configuration data seeded successfully")
        
        # Close engine
        engine.dispose()
        
        logger.info("=" * 80)
        logger.info("✅ DATABASE INITIALIZATION COMPLETE")
        logger.info("=" * 80)
    except Exception as error:
        logger.error(f"❌ Database initialization failed: {error}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
