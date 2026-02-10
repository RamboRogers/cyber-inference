"""
Database management for Cyber-Inference.

Provides:
- SQLAlchemy async engine and session management
- Database initialization and migrations
- Connection pooling
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# Global engine and session factory
_engine: Optional[create_async_engine] = None
_session_factory: Optional[async_sessionmaker] = None


async def init_database(db_path: Path) -> None:
    """
    Initialize the database connection and create tables.

    Args:
        db_path: Path to the SQLite database file
    """
    global _engine, _session_factory

    logger.info(f"[info]Initializing database at: {db_path}[/info]")

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create async engine
    db_url = f"sqlite+aiosqlite:///{db_path}"
    _engine = create_async_engine(
        db_url,
        echo=False,  # Set to True for SQL query logging
        pool_pre_ping=True,
    )

    logger.debug(f"Database engine created: {db_url}")

    # Enable foreign keys for SQLite
    @event.listens_for(_engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
        logger.debug("SQLite pragmas set: foreign_keys=ON, journal_mode=WAL")

    # Create session factory
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Import models to ensure they're registered
    from cyber_inference.models import db_models  # noqa: F401

    # Create all tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("[success]Database tables created/verified[/success]")

    # Migrate existing tables: add any missing columns (SQLite CREATE TABLE
    # IF NOT EXISTS won't add new columns to tables that already exist).
    await _migrate_add_missing_columns()

    # Log table info
    for table_name in Base.metadata.tables.keys():
        logger.debug(f"  Table registered: {table_name}")


async def _migrate_add_missing_columns() -> None:
    """Add columns defined in ORM models but missing from existing SQLite tables."""
    import sqlalchemy as sa

    async with _engine.begin() as conn:
        for table_name, table in Base.metadata.tables.items():
            # Get existing column names from SQLite
            existing = await conn.execute(sa.text(f"PRAGMA table_info('{table_name}')"))
            existing_cols = {row[1] for row in existing.fetchall()}

            for column in table.columns:
                if column.name not in existing_cols:
                    # Build ALTER TABLE ADD COLUMN statement
                    col_type = column.type.compile(dialect=_engine.dialect)
                    nullable = "" if column.nullable else " NOT NULL"
                    default = ""
                    if column.default is not None:
                        val = column.default.arg
                        if isinstance(val, str):
                            default = f" DEFAULT '{val}'"
                        elif val is not None:
                            default = f" DEFAULT {val}"
                    stmt = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type}{nullable}{default}"
                    try:
                        await conn.execute(sa.text(stmt))
                        logger.info(f"  Migration: added column {table_name}.{column.name}")
                    except Exception as e:
                        # Column may already exist (race condition) or other issue
                        logger.debug(f"  Migration skip {table_name}.{column.name}: {e}")


async def close_database() -> None:
    """Close the database connection."""
    global _engine

    if _engine:
        logger.info("[info]Closing database connection[/info]")
        await _engine.dispose()
        _engine = None
        logger.debug("Database connection closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database sessions.

    Yields:
        AsyncSession: Database session for the request
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _session_factory() as session:
        try:
            logger.debug("Database session opened")
            yield session
            await session.commit()
            logger.debug("Database session committed")
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            logger.debug("Database session closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for getting database sessions outside of FastAPI.

    Usage:
        async with get_db_session() as session:
            result = await session.execute(...)
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_engine():
    """Get the database engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized")
    return _engine

