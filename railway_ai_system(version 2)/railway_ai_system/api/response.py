"""
RailDrishti AI — Autonomous Response Dispatcher
================================================
Fires within 2 seconds of a TBE tamper_confidence >= 75% alarm.
No human operator required in the first response loop.

Pillars:
  1. SMS/WhatsApp to nearest RPF patrol (MSG91 / Twilio)
  2. On-site PA/Siren webhook trigger (Edge GPIO)
  3. KAVACH Speed Restriction Advisory (stub — requires Railway Board API key)
  4. Response log to DB for audit trail
"""

import os
import time
import json
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import api.db as db

logger = logging.getLogger("raildrishti.response")

# ──────────────────────────────────────────────────────────
#  CONFIGURATION  (set real values via environment variables)
# ──────────────────────────────────────────────────────────
MSG91_AUTH_KEY    = os.getenv("MSG91_AUTH_KEY", "")       # MSG91 API key
MSG91_SENDER_ID   = os.getenv("MSG91_SENDER_ID", "RLDRAI")
MSG91_TEMPLATE_ID = os.getenv("MSG91_TEMPLATE_ID", "")
RPF_PHONE_NUMBERS = os.getenv("RPF_PHONES", "").split(",") # comma-separated numbers

TWILIO_SID        = os.getenv("TWILIO_SID", "")
TWILIO_TOKEN      = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM       = os.getenv("TWILIO_FROM", "")

SIREN_WEBHOOK_URL = os.getenv("SIREN_WEBHOOK_URL", "")    # Edge PA/GPIO endpoint
KAVACH_API_URL    = os.getenv("KAVACH_API_URL", "")       # Railway Board API (future)
KAVACH_API_KEY    = os.getenv("KAVACH_API_KEY", "")

DEMO_MODE = os.getenv("RAILDRISHTI_DEMO", "true").lower() == "true"

# ──────────────────────────────────────────────────────────
#  RPF NEAREST STATION LOOKUP
#  In production: replace with a real GIS database query
# ──────────────────────────────────────────────────────────
RPF_PATROL_MAP = {
    "Delhi":      "+911234567890",
    "Mumbai":     "+911234567891",
    "Chennai":    "+911234567892",
    "Kolkata":    "+911234567893",
    "Bengaluru":  "+911234567894",
    "Hyderabad":  "+911234567895",
    "Jaipur":     "+911234567896",
    "Lucknow":    "+911234567897",
    "Patna":      "+911234567898",
    "Bhopal":     "+911234567899",
    # ... extend with all 68 divisional RPF numbers
}


def _format_sms(incident: dict) -> str:
    """Build a crisp, actionable SMS body for RPF."""
    conf  = incident.get("tamper_confidence", "?")
    city  = incident.get("city",  incident.get("zone", "Unknown"))
    cr    = incident.get("control_room", "CR-?")
    cam   = incident.get("sensor", "Cam-?")
    lat   = incident.get("lat", "")
    lng   = incident.get("lng", "")
    ts    = datetime.fromtimestamp(incident.get("timestamp", time.time())).strftime("%H:%M:%S")
    maps  = f"https://maps.google.com/maps?q={lat},{lng}" if lat and lng else "N/A"

    return (
        f"[RailDrishti AI ALERT] {ts}\n"
        f"TRACK TAMPERING DETECTED\n"
        f"Location: {city} > {cr} > {cam}\n"
        f"Confidence: {conf}%\n"
        f"GPS: {maps}\n"
        f"Action: Dispatch patrol IMMEDIATELY."
    )


# ──────────────────────────────────────────────────────────
#  RESPONSE ACTIONS
# ──────────────────────────────────────────────────────────
def _send_sms_msg91(phone: str, message: str) -> dict:
    """Send SMS via MSG91 API."""
    if not MSG91_AUTH_KEY:
        return {"provider": "msg91", "status": "skipped", "reason": "no_api_key"}
    try:
        resp = requests.post(
            "https://api.msg91.com/api/v5/flow/",
            headers={"authkey": MSG91_AUTH_KEY, "Content-Type": "application/json"},
            json={
                "template_id": MSG91_TEMPLATE_ID,
                "sender":      MSG91_SENDER_ID,
                "mobiles":     phone.replace("+", ""),
                "VAR1":        message[:160]
            },
            timeout=5
        )
        return {"provider": "msg91", "status": "sent", "code": resp.status_code}
    except Exception as e:
        return {"provider": "msg91", "status": "error", "error": str(e)}


def _send_sms_twilio(phone: str, message: str) -> dict:
    """Fallback: send SMS via Twilio."""
    if not TWILIO_SID:
        return {"provider": "twilio", "status": "skipped", "reason": "no_api_key"}
    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            auth=(TWILIO_SID, TWILIO_TOKEN),
            data={"From": TWILIO_FROM, "To": phone, "Body": message},
            timeout=5
        )
        return {"provider": "twilio", "status": "sent", "code": resp.status_code}
    except Exception as e:
        return {"provider": "twilio", "status": "error", "error": str(e)}


def _trigger_siren(incident: dict) -> dict:
    """POST to edge siren/PA webhook. In demo mode, just logs."""
    city = incident.get("city", "Unknown")
    cr   = incident.get("control_room", "CR-1")

    if DEMO_MODE or not SIREN_WEBHOOK_URL:
        logger.info(f"[SIREN DEMO] Would trigger PA at {city} / {cr}")
        return {"action": "siren", "status": "demo_logged", "location": f"{city}/{cr}"}

    try:
        resp = requests.post(
            SIREN_WEBHOOK_URL,
            json={"location": f"{city}/{cr}", "duration_seconds": 15, "type": "tamper_alert"},
            timeout=3
        )
        return {"action": "siren", "status": "triggered", "code": resp.status_code}
    except Exception as e:
        return {"action": "siren", "status": "error", "error": str(e)}


def _kavach_advisory(incident: dict) -> dict:
    """
    Send a KAVACH Speed Restriction Advisory.
    Schema matches KAVACH's documented SoA messaging format.
    Requires Railway Board API credentials — stub in demo mode.
    """
    lat   = incident.get("lat", "28.6")
    lng   = incident.get("lng", "77.2")
    level = incident.get("level", "warning")

    payload = {
        "advisory_type": "SPEED_RESTRICTION",
        "severity":      "EMERGENCY" if level == "critical" else "CAUTION",
        "location": {"lat": lat, "lng": lng, "radius_km": 5},
        "max_speed_kmh": 0 if level == "critical" else 30,
        "reason":        "Track tampering detected by RailDrishti AI",
        "issued_at":     datetime.utcnow().isoformat(),
        "source":        "RAILDRISHTI_AI_v1"
    }

    if DEMO_MODE or not KAVACH_API_URL:
        logger.info(f"[KAVACH DEMO] Advisory payload: {json.dumps(payload)}")
        return {"action": "kavach", "status": "demo_logged", "payload": payload}

    try:
        resp = requests.post(
            KAVACH_API_URL,
            headers={"X-API-Key": KAVACH_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=5
        )
        return {"action": "kavach", "status": "sent", "code": resp.status_code}
    except Exception as e:
        return {"action": "kavach", "status": "error", "error": str(e)}


# ──────────────────────────────────────────────────────────
#  MAIN DISPATCHER — called by TBE alarm trigger
# ──────────────────────────────────────────────────────────
def dispatch(incident: dict) -> list:
    """
    Fire all autonomous responses in sequence.
    Returns a list of response result dicts for audit logging.
    incident: the same dict pushed to the WebSocket alarm queue
    """
    actions = []
    city    = incident.get("city", incident.get("zone", ""))
    message = _format_sms(incident)

    # 1. SMS to RPF
    # Priority: use RPF_PHONES from .env (your verified number), then fall back to zone map
    phones_to_alert = [p.strip() for p in RPF_PHONE_NUMBERS if p.strip()] or \
                      ([RPF_PATROL_MAP[city]] if city in RPF_PATROL_MAP else [])

    if phones_to_alert:
        for phone in phones_to_alert:
            result = _send_sms_msg91(phone, message)
            if result.get("status") in ("skipped", "error"):
                result = _send_sms_twilio(phone, message)
            actions.append(result)
            logger.info(f"[SMS] Sent to {phone}: {result}")
    else:
        logger.info(f"[SMS DEMO] {message}")
        actions.append({"action": "sms", "status": "demo_logged", "city": city})

    # 2. On-site siren
    actions.append(_trigger_siren(incident))

    # 3. KAVACH advisory
    actions.append(_kavach_advisory(incident))

    # Audit log to DB
    try:
        incident_id = incident.get("id")
        if incident_id:
            db.log_response_actions(incident_id, actions)
    except Exception:
        pass

    logger.info(f"[DISPATCH] Incident #{incident.get('id')} — {len(actions)} actions fired.")
    return actions
