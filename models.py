from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text

from backend_api.db import Base


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(String(300), nullable=False)
    severity = Column(String(10), nullable=False)  # high/medium/low

    pred_label = Column(String(80), nullable=True)
    confidence = Column(Float, nullable=True)
    flow = Column(Integer, nullable=True)

    details_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
