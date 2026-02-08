from __future__ import annotations

import json
from datetime import datetime
from sqlalchemy.orm import Session

from backend_api.models import Alert


def severity_from_confidence(conf: float) -> str:
    if conf >= 0.85:
        return "high"
    if conf >= 0.65:
        return "medium"
    return "low"


def create_alert_from_event(db: Session, event: dict) -> Alert:
    conf = float(event.get("confidence", 0.0))
    pred_label = event.get("pred_label", "Unknown")

    msg = f"{pred_label} detected (conf={conf:.3f})"
    sev = severity_from_confidence(conf)

    alert = Alert(
        message=msg,
        severity=sev,
        pred_label=pred_label,
        confidence=conf,
        flow=event.get("flow"),
        details_json=json.dumps(event.get("details", {}), ensure_ascii=False),
        resolved=False,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def list_alerts(
    db: Session,
    search: str | None = None,
    severity: str | None = None,
    resolved: str | None = "false",
    limit: int = 200,
):
    # ✅ show newest first
    q = db.query(Alert).order_by(Alert.created_at.desc())

    if search:
        q = q.filter(Alert.message.ilike(f"%{search}%"))

    if severity in ("high", "medium", "low"):
        q = q.filter(Alert.severity == severity)

    if resolved == "true":
        q = q.filter(Alert.resolved.is_(True))
    elif resolved == "false":
        q = q.filter(Alert.resolved.is_(False))

    return q.limit(limit).all()


def resolve_alert(db: Session, alert_id: int) -> Alert | None:
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        return None
    alert.resolved = True
    alert.resolved_at = datetime.utcnow()
    db.commit()
    db.refresh(alert)
    return alert
