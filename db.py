from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DB_USER = os.getenv("SAFEDOSE_DB_USER", "safedose_user")
DB_PASS = os.getenv("SAFEDOSE_DB_PASS", "StrongPassword123!")
DB_HOST = os.getenv("SAFEDOSE_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("SAFEDOSE_DB_PORT", "3306"))
DB_NAME = os.getenv("SAFEDOSE_DB_NAME", "safedose")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,  # 30 min (helps with XAMPP idle disconnect)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
