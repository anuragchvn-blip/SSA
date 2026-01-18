"""Database session management and connection handling."""

from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import logging

from src.core.config import settings
from src.core.exceptions import DatabaseError
from src.data.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections and sessions."""
    
    def __init__(self):
        self._engine = None
        self._session_local = None
        self._initialized = False
    
    def initialize(self):
        """Initialize database engine and session factory."""
        if self._initialized:
            return
            
        try:
            # Determine if we're using SQLite
            db_uri = settings.database.sqlalchemy_database_uri
            is_sqlite = db_uri.startswith('sqlite:')
            
            # Create engine with appropriate settings
            if is_sqlite:
                # SQLite settings
                self._engine = create_engine(
                    db_uri,
                    poolclass=StaticPool,  # SQLite works best with static pool
                    connect_args={"check_same_thread": False},  # Required for threading
                    echo=settings.debug,
                )
            else:
                # PostgreSQL settings
                self._engine = create_engine(
                    db_uri,
                    poolclass=QueuePool,
                    pool_size=20,
                    max_overflow=30,
                    pool_pre_ping=True,  # Verify connections before use
                    pool_recycle=3600,   # Recycle connections after 1 hour
                    echo=settings.debug,
                )
            
            # Configure connection events for monitoring
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                """Set SQLite pragmas for better performance."""
                if 'sqlite' in settings.database.sqlalchemy_database_uri:
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.close()
            
            @event.listens_for(self._engine, "checkout")
            def receive_checkout(dbapi_connection, connection_record, connection_proxy):
                """Log connection checkout for monitoring."""
                logger.debug("Database connection checked out", 
                           connection_id=id(connection_record))
            
            # Create session factory
            self._session_local = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine
            )
            
            self._initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database manager", extra={"error": str(e)})
            raise DatabaseError(
                message="Database initialization failed",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e)}
            )
    
    @property
    def engine(self):
        """Get database engine."""
        if not self._initialized:
            self.initialize()
        return self._engine
    
    @property
    def session_local(self):
        """Get session factory."""
        if not self._initialized:
            self.initialize()
        return self._session_local
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        if not self._initialized:
            self.initialize()
            
        session = self._session_local()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Database transaction failed", extra={"error": str(e)})
            raise DatabaseError(
                message="Database transaction failed",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e)}
            )
        except Exception as e:
            session.rollback()
            logger.error("Unexpected error in database session", extra={"error": str(e)})
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables."""
        if not self._initialized:
            self.initialize()
            
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", extra={"error": str(e)})
            raise DatabaseError(
                message="Failed to create database tables",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e)}
            )
    
    def drop_tables(self):
        """Drop all database tables (for testing)."""
        if not self._initialized:
            self.initialize()
            
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error("Failed to drop database tables", extra={"error": str(e)})
            raise DatabaseError(
                message="Failed to drop database tables",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e)}
            )
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        if not self._initialized:
            return False
            
        try:
            from sqlalchemy import text
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {str(e)}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    with db_manager.get_session() as session:
        yield session


def init_db():
    """Initialize database for application startup."""
    try:
        db_manager.initialize()
        db_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        # Raise the exception to prevent silent failures
        raise


def close_db():
    """Close database connections for application shutdown."""
    # Engine cleanup happens automatically
    pass