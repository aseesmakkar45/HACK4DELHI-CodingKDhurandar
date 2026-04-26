from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from server.database import get_db
from server.models import Alert

router = APIRouter()

@router.get("/")
def get_alerts(db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(desc(Alert.timestamp)).all()
    return alerts

@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert:
        alert.acknowledged = True
        db.commit()
    return {"status": "acknowledged", "alert_id": alert_id}
