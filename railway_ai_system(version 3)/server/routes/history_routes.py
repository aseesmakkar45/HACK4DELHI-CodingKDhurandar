from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from server.database import get_db
from server.models import Run

router = APIRouter()

@router.get("/")
def get_history(db: Session = Depends(get_db)):
    runs = db.query(Run).order_by(desc(Run.timestamp)).all()
    return runs
