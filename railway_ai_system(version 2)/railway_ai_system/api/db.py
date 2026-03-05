import sqlite3
import os
import time
import json
from contextlib import contextmanager

# Define where the db lives. We'll put it in the api folder
DB_FILE = os.path.join(os.path.dirname(__file__), "railway_system.db")

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                level TEXT NOT NULL,
                zone TEXT NOT NULL,
                sensor TEXT NOT NULL,
                timestamp REAL NOT NULL,
                status TEXT DEFAULT 'active',
                operator_comment TEXT DEFAULT ''
            )
        ''')
        try:
            cursor.execute("ALTER TABLE incidents ADD COLUMN resolved_at REAL DEFAULT NULL")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE incidents ADD COLUMN region TEXT DEFAULT 'Northern India'")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE incidents ADD COLUMN city TEXT DEFAULT 'Delhi'")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE incidents ADD COLUMN control_room TEXT DEFAULT 'CR-1'")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE incidents ADD COLUMN lat TEXT DEFAULT '28.6139'")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE incidents ADD COLUMN lng TEXT DEFAULT '77.2090'")
        except sqlite3.OperationalError: pass
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS response_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                actions_json TEXT NOT NULL,
                fired_at REAL NOT NULL
            )
        ''')
        conn.commit()

def log_incident(message, level, zone, sensor, timestamp, region="Northern India", city="Delhi", control_room="CR-1", lat="28.6", lng="77.2"):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO incidents (message, level, zone, sensor, timestamp, status, region, city, control_room, lat, lng) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (message, level, zone, sensor, timestamp, 'active', region, city, control_room, lat, lng)
        )
        conn.commit()
        return cursor.lastrowid

def get_active_incident(sensor: str):
    """Return the ID of an existing active (unresolved) incident for a given sensor, or None."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM incidents WHERE sensor = ? AND status = 'active' ORDER BY timestamp DESC LIMIT 1",
            (sensor,)
        )
        row = cursor.fetchone()
        return row["id"] if row else None

def update_incident_confidence(incident_id: int, message: str, tamper_conf: int):
    """Update an existing incident's message and confidence when the event is still ongoing."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE incidents SET message = ?, level = CASE WHEN level = 'critical' THEN 'critical' ELSE level END WHERE id = ?",
            (message, incident_id)
        )
        conn.commit()

def get_incidents(zone=None):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if zone and zone != "All Zones":
            cursor.execute('SELECT * FROM incidents WHERE zone = ? ORDER BY timestamp DESC LIMIT 100', (zone,))
        else:
            cursor.execute('SELECT * FROM incidents ORDER BY timestamp DESC LIMIT 100')
        return [dict(row) for row in cursor.fetchall()]

def acknowledge_incident(incident_id, comment):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE incidents SET status = ?, operator_comment = ?, resolved_at = ? WHERE id = ?',
            ('resolved', comment, time.time(), incident_id)
        )
        conn.commit()

def log_response_actions(incident_id: int, actions: list):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO response_log (incident_id, actions_json, fired_at) VALUES (?, ?, ?)',
            (incident_id, json.dumps(actions), time.time())
        )
        conn.commit()

def get_response_log(incident_id: int):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM response_log WHERE incident_id = ? ORDER BY fired_at DESC', (incident_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def clear_incidents():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM incidents')
        cursor.execute('DELETE FROM response_log')
        cursor.execute('UPDATE sqlite_sequence SET seq = 0 WHERE name IN ("incidents", "response_log")')
        conn.commit()

# Ensure tables exist at startup
init_db()
