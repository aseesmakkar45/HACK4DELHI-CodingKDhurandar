from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from server.database import Base
import datetime

class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, index=True) # run_id
    video_name = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String) # processing, complete, failed
    risk_score = Column(Float, default=0.0)
    anomalies_detected = Column(Integer, default=0)
    humans_detected = Column(Integer, default=0)
    total_frames = Column(Integer, default=0)
    
    alerts = relationship("Alert", back_populates="run")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, index=True) # uuid
    run_id = Column(String, ForeignKey("runs.id"))
    message = Column(String)
    severity = Column(String) # high, medium, low
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    acknowledged = Column(Boolean, default=False)

    run = relationship("Run", back_populates="alerts")
