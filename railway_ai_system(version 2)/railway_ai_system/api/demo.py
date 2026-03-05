"""
RailDrishti AI — Pitch Demo Mode
=================================
Simulates realistic tamper incidents through the real WebSocket + DB pipeline.
Activate by setting env var: RAILDRISHTI_DEMO_MODE=true
Or by calling POST /api/demo/start

No camera required. Broadcasts real-looking incident streams for presentations.
"""

import asyncio
import random
import time
import logging

import api.db as db
import api.response as response_dispatcher

logger = logging.getLogger("raildrishti.demo")

_REGIONS = ['Northern India', 'Southern India', 'Western India', 'Eastern India', 'Central India']
_CITIES = {
    'Northern India': ['Delhi', 'Chandigarh', 'Jaipur', 'Lucknow', 'Kanpur'],
    'Southern India': ['Chennai', 'Bengaluru', 'Hyderabad', 'Kochi', 'Madurai'],
    'Western India':  ['Mumbai', 'Ahmedabad', 'Pune', 'Surat', 'Nagpur'],
    'Eastern India':  ['Kolkata', 'Patna', 'Bhubaneswar', 'Guwahati', 'Ranchi'],
    'Central India':  ['Bhopal', 'Indore', 'Raipur', 'Jabalpur', 'Gwalior']
}

# Realistic tamper event descriptions modelled on actual RPF FIR language
_TAMPER_MESSAGES = [
    "Person crouching at fishplate junction with metallic tool. TBE score sustained above threshold for 8s.",
    "Two individuals using angle grinder on rail joint. Continuous stationary posture detected.",
    "Suspected spike removal activity — wrist keypoints at track level for >6 seconds.",
    "Individual bending at rail clip with tool. Body pose indicates sustained contact with track hardware.",
    "Unidentified person removing ballast stones from track bed. Low crouch + stationary for 12s.",
    "Rail bolt unscrewing detected — wrist and elbow keypoints locked at track level.",
    "Signal cable interference suspected. Person crouched at signal post base for >10s.",
    "Track intrusion: crowbar-like object detected in proximity to fishplate. Alarm threshold exceeded.",
]

_RESOLUTION_MESSAGES = [
    "RPF patrol dispatched. Suspects fled on approach. FIR #RLY-2024-0{} filed.",
    "Ground team inspected. Track integrity confirmed. Preventive action taken.",
    "Loco Pilot alerted via Control Room. Train decelerated. Obstruction cleared.",
    "Section patrolling increased. CCTV evidence secured for investigation.",
]

demo_running = False
alarm_queue_ref = None
main_loop_ref = None


async def _generate_incident_stream(interval_seconds: float = 15):
    """Fire simulated tamper incidents every `interval_seconds`."""
    global demo_running
    incident_count = 0

    logger.info("[DEMO] Starting incident stream...")

    while demo_running:
        await asyncio.sleep(interval_seconds)

        if not demo_running:
            break

        region = random.choice(_REGIONS)
        city   = random.choice(_CITIES[region])
        cr     = f"CR-{random.randint(1, 5)}"
        sensor = f"Camera-{random.randint(100, 999)}"
        lat    = str(round(20.0 + random.uniform(-5, 10), 4))
        lng    = str(round(78.0 + random.uniform(-5, 10), 4))

        confidence = random.randint(76, 98)
        level      = "critical" if confidence >= 90 else "warning"
        dist       = random.randint(1, 12)
        direction  = random.choice(["approaching", "departing"])
        action     = "STOP TRAIN IMMEDIATELY" if (level == "critical" and direction == "approaching") else "Track Inspection Required"

        msg = (
            f"{random.choice(_TAMPER_MESSAGES)} "
            f"Train {direction} at {dist}km. Action: {action}"
        )

        current_time = time.time()
        incident_id  = db.log_incident(
            msg, level, city, sensor, current_time,
            region, city, cr, lat, lng
        )

        alarm_payload = {
            "id": incident_id, "type": "alert", "message": msg,
            "timestamp": current_time, "level": level,
            "zone": city, "region": region, "city": city,
            "control_room": cr, "sensor": sensor,
            "lat": lat, "lng": lng,
            "tamper_confidence": confidence,
            "status": "active",
            "demo": True
        }

        if alarm_queue_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(alarm_queue_ref.put_nowait, alarm_payload)

        # Fire autonomous response simulation
        response_dispatcher.dispatch(alarm_payload)

        incident_count += 1
        logger.info(f"[DEMO] Incident #{incident_id} fired (conf={confidence}%, {city})")

        # Simulate resolution after ~30s for some incidents
        if random.random() > 0.5:
            await asyncio.sleep(30)
            resolution = random.choice(_RESOLUTION_MESSAGES).format(
                str(incident_id).zfill(4)
            )
            db.acknowledge_incident(incident_id, f"[AUTO-RESOLVED DEMO] {resolution}")

    logger.info(f"[DEMO] Stream stopped. {incident_count} incidents generated.")


def start_demo(alarm_queue, main_loop):
    """Start the demo mode stream. Called from main.py on API trigger."""
    global demo_running, alarm_queue_ref, main_loop_ref
    demo_running    = True
    alarm_queue_ref = alarm_queue
    main_loop_ref   = main_loop
    asyncio.ensure_future(_generate_incident_stream(interval_seconds=12))
    logger.info("[DEMO] Demo mode activated.")


def stop_demo():
    global demo_running
    demo_running = False
    logger.info("[DEMO] Demo mode deactivated.")
