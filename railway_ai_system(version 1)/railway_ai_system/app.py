
import os
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
import datetime


# ZONES + USERS + SENSORS CONFIG

ZONES_CONFIG = {
    "North Delhi": {
        "username": "north_admin",
        "password": "north123",
        "sensors": 5
    },
    "South Delhi": {
        "username": "south_admin",
        "password": "south123",
        "sensors": 8
    },
    "West Delhi": {
        "username": "west_admin",
        "password": "west123",
        "sensors": 4
    },
    "East Delhi": {
        "username": "east_admin",
        "password": "east123",
        "sensors": 6
    },
    "Central Delhi": {
        "username": "central_admin",
        "password": "central123",
        "sensors": 5
    }
}

ZONE_SENSORS = {
    "North Delhi": {
        "Sensor 1": {"name": "Kashmere Gate", "lat": 28.6675, "lon": 77.2281},
        "Sensor 2": {"name": "Civil Lines", "lat": 28.6764, "lon": 77.2247},
        "Sensor 3": {"name": "Model Town", "lat": 28.7167, "lon": 77.1910},
        "Sensor 4": {"name": "Burari", "lat": 28.7480, "lon": 77.2000},
        "Sensor 5": {"name": "Wazirabad", "lat": 28.7066, "lon": 77.2387},
    },

    "South Delhi": {
        "Sensor 1": {"name": "Saket", "lat": 28.5245, "lon": 77.2066},
        "Sensor 2": {"name": "Hauz Khas", "lat": 28.5494, "lon": 77.2001},
        "Sensor 3": {"name": "Malviya Nagar", "lat": 28.5355, "lon": 77.2090},
        "Sensor 4": {"name": "Kalkaji", "lat": 28.5352, "lon": 77.2597},
        "Sensor 5": {"name": "Mehrauli", "lat": 28.5246, "lon": 77.1855},
        "Sensor 6": {"name": "Chhatarpur", "lat": 28.4986, "lon": 77.1650},
        "Sensor 7": {"name": "Lajpat Nagar", "lat": 28.5672, "lon": 77.2431},
        "Sensor 8": {"name": "Defence Colony", "lat": 28.5733, "lon": 77.2300},
    },

    "West Delhi": {
        "Sensor 1": {"name": "Janakpuri", "lat": 28.6219, "lon": 77.0878},
        "Sensor 2": {"name": "Uttam Nagar", "lat": 28.6210, "lon": 77.0600},
        "Sensor 3": {"name": "Dwarka", "lat": 28.5921, "lon": 77.0460},
        "Sensor 4": {"name": "Punjabi Bagh", "lat": 28.6683, "lon": 77.1334},
    },

    "East Delhi": {
        "Sensor 1": {"name": "Laxmi Nagar", "lat": 28.6370, "lon": 77.2773},
        "Sensor 2": {"name": "Preet Vihar", "lat": 28.6415, "lon": 77.2950},
        "Sensor 3": {"name": "Mayur Vihar", "lat": 28.6040, "lon": 77.2890},
        "Sensor 4": {"name": "Anand Vihar", "lat": 28.6469, "lon": 77.3160},
        "Sensor 5": {"name": "Vivek Vihar", "lat": 28.6720, "lon": 77.3150},
        "Sensor 6": {"name": "Shahdara", "lat": 28.6733, "lon": 77.2890},
    },

    "Central Delhi": {
        "Sensor 1": {"name": "Connaught Place", "lat": 28.6315, "lon": 77.2167},
        "Sensor 2": {"name": "Karol Bagh", "lat": 28.6517, "lon": 77.1900},
        "Sensor 3": {"name": "Paharganj", "lat": 28.6440, "lon": 77.2150},
        "Sensor 4": {"name": "Rajiv Chowk", "lat": 28.6328, "lon": 77.2197},
        "Sensor 5": {"name": "ITO", "lat": 28.6280, "lon": 77.2410},
    },
}



# PAGE CONFIG (APP LOOK & FEEL)

# Ye block app ko website / dashboard
# jaisa feel dene ke liye use hota hai
st.set_page_config(
    page_title="AI Railway Safety System",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)



# LOAD CUSTOM CSS (OPTIONAL)

def load_css():
    # CSS file ko load karta hai agar available ho
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")

    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ style.css not found, running without custom CSS")

load_css()

# =========================
# SESSION STATE INIT 
# =========================
if "incident_logs" not in st.session_state:
    st.session_state.incident_logs = []

if "alert_acknowledged" not in st.session_state:
    st.session_state.alert_acknowledged = False

if "operator_note" not in st.session_state:
    st.session_state.operator_note = ""


# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "active_zone" not in st.session_state:
    st.session_state.active_zone = None    

# =========================
# WELCOME / LOGIN GATE
# =========================
if not st.session_state.logged_in:
    st.markdown("""
        <div style="
            background-color:#ffffff;
            color:#e5e7eb;
            padding:8px 16px;
            font-size:13px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            border-bottom:1px solid #1f2933;
        ">
            <div>
                🏛️ <b>Government of India</b> | Railway Safety & Infrastructure Monitoring
            </div>
            <div>
                🔐 Authorized Access Only &nbsp; | &nbsp; 🕒 <span id="time"></span>
            </div>
        </div>

        <script>
        setInterval(() => {
        const now = new Date();
        document.getElementById("time").innerText =
            now.toLocaleDateString() + " " + now.toLocaleTimeString();
        }, 1000);
        </script>
        """, unsafe_allow_html=True)

    home_tab, run_tab, about_tab, contact_tab = st.tabs(
    ["🏠 HOME", "▶️ RUN ANALYSIS", "ℹ️ ABOUT US", "📞 CONTACT US"]
)
    
    with home_tab:
        st.image(
        "images/banner.png",
        use_container_width=True
    )

        st.markdown("""
        <style>
        img {
            max-height: 260px;   /* 👈 YAHAN HEIGHT CONTROL */
            object-fit: cover;
        }
        </style>
        """, unsafe_allow_html=True)
            

        st.markdown("""
        # 🏛️ AI‑Enabled Railway Tampering Detection & Safety Monitoring Portal  
        ### Government of India | Smart Railway Infrastructure Initiative
        """)

        st.markdown("---")

        # =========================
        # PROBLEM STATEMENT
        # =========================
        st.markdown("""
        
        Indian Railways operates a vast and geographically distributed railway network that is continuously exposed to **tampering‑related threats**, including **track interference, vandalism, sabotage, theft of railway assets, and negligence‑induced damage**.

        These activities, whether intentional or unintentional, pose **serious risks to passenger safety, train operations, and national infrastructure security**. Existing monitoring mechanisms rely heavily on **manual inspections and isolated surveillance systems**, resulting in delayed detection and reactive response.

        The absence of an **integrated, real‑time, tampering‑focused monitoring platform** limits the ability of authorities to proactively identify threats, correlate sensor data with evidence, and ensure structured accountability.
        """)

        st.markdown("---")

        # =========================
        # NEED FOR SOLUTION
        # =========================
        st.markdown("""
        ## Need for a Government‑Grade Solution

        There is a critical requirement for a **secure, centralized, and governance‑ready digital platform** capable of detecting tampering‑related anomalies in real time, supporting **human‑in‑the‑loop decision making**, and maintaining **zone‑wise operational independence**.

        Such a system must be specifically designed for **government operations**, prioritizing reliability, transparency, auditability, and controlled response mechanisms.
        """)
        col_img, col_text = st.columns([1, 2])  # 👈 left small, right large

        with col_img:
            st.image(
                "images/collage.png",  # 👈 image ka local path
                use_container_width=True
            )

        with col_text:
            st.markdown("""
            Large‑scale railway infrastructure remains vulnerable to organized and repeated
            tampering attempts, including removal of track components, deliberate obstruction,
            and coordinated sabotage.

            Such incidents, often detected only after significant delay, highlight the
            limitations of manual patrolling and fragmented monitoring mechanisms. The scale,
            frequency, and severity of these threats necessitate a **centralized, government‑led,
            technology‑driven monitoring and decision‑support system** to ensure passenger safety
            and infrastructure integrity.
            """)




        st.markdown("---")

        # =========================
        # PURPOSE
        # =========================
        st.markdown("""
        ## Purpose of the Portal

        The purpose of this portal is to function as a **centralized AI‑assisted monitoring and decision‑support system** for detecting, verifying, and managing **tampering‑related risks** across railway infrastructure.

        The platform aims to:
        - Enhance passenger safety and operational reliability  
        - Enable early detection of infrastructure tampering  
        - Generate timely alerts for operational authorities  
        - Support structured incident documentation and evidence management  
        """)

        st.markdown("---")

        # =========================
        # ROLE OF AI
        # =========================
        col_text, col_img = st.columns([2, 1])  # 👈 left text zyada, right image chhoti

    with col_text:
        st.markdown("""
        ## Role of Artificial Intelligence

        Artificial Intelligence within the system functions as an **assisted intelligence layer**, supporting operators by identifying **patterns indicative of tampering, abnormal sensor behavior, and system degradation**.

        AI is utilized for:
        - Assisting in early anomaly detection  
        - Supporting sensor health assessment  
        - Enabling intelligent alert prioritization  

        All critical decisions remain under **human oversight**, ensuring compliance with safety‑critical governance requirements.
        """)
        with col_img:
            st.image(
        "images/controlroom.png",  # 👈 apni image ka exact naam yahan
        use_container_width=True
            )


        st.markdown("---")
        # =========================
        # KEY CAPABILITIES
        # =========================
        st.markdown("""
        ## Key Capabilities

        - Zone‑based secure access with operational isolation  
        - Tampering‑focused surveillance using multi‑sensor inputs  
        - Real‑time alerts with mandatory operator acknowledgement  
        - Evidence‑linked incident management (CCTV, drone, audio)  
        - Sensor‑wise system health monitoring  
        - Geographic visualization of sensor locations  
        - Audit‑ready, zone‑isolated incident history  
        """)

        st.markdown("---")

        # =========================
        # VISION
        # =========================
        st.markdown("""
        ## Vision

        The portal is envisioned as a **national‑grade digital safety backbone** for railway operations, enabling authorities to transition from **reactive incident handling to proactive tampering prevention**, supported by **AI‑assisted intelligence and structured governance frameworks**.
        """)

        st.markdown("")  # thoda spacing

        # 👇 CENTERED IMAGE
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            st.image(
                "images/delhi.png",  # 👈 apni image ka exact naam
                use_container_width=True
            )

        st.markdown("---")

        st.success(
            "This portal is designed for authorized government personnel and supports secure, zone‑isolated railway safety operations."
        )


    with run_tab:
        st.title("🚆 Railway Safety System – Login")

        selected_zone = st.selectbox(
            "Select Railway Zone",
            list(ZONES_CONFIG.keys())
        )

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            zone_data = ZONES_CONFIG[selected_zone]

            if username == zone_data["username"] and password == zone_data["password"]:
                st.session_state.logged_in = True
                st.session_state.active_zone = selected_zone
                st.success(f"Logged in to {selected_zone}")
                st.rerun()
            else:
                st.error("Invalid credentials for selected zone")

    with about_tab:
        st.markdown("## ℹ️ About Us")

        st.write("""
        **AI Railway Safety System** is a smart monitoring platform
        developed to enhance railway track safety using
        multi‑sensor intelligence and automated decision making.
        """)

        st.markdown("### 🎯 Our Objective")
        st.markdown("""
        - Prevent railway accidents  
        - Detect track tampering in real‑time  
        - Assist control rooms with fast decisions  
        """)

        st.markdown("### 🧠 Technology Stack")
        st.markdown("""
        - Machine Learning & Signal Processing  
        - Computer Vision (CCTV & Drone feeds)  
        - Acoustic & Vibration Analysis  
        - Real‑time Decision Engine  
        """)
    with contact_tab:
        st.markdown("## 📞 Contact Us")

        st.write("""
        For system access, collaboration, or technical queries,
        please reach out to the development team.
        """)

        st.markdown("### 📧 Email")
        st.write("railwaysafety.ai@gmail.com")

        st.markdown("### 📍 Organization")
        st.write("Railway Safety Innovation Lab")

        st.markdown("### 🛠 Support")
        st.write("Available during demo and evaluation sessions.")

    st.stop()

# =====================================================
# IMPORT BACKEND MODULES
# =====================================================
from backend.vibration import analyze_vibration
from backend.vision import analyze_visual
from backend.sound import analyze_sound
from backend.weather import fetch_weather_forecast
from backend.train_schedule import get_train_status, get_direction
from backend.maintenance_db import is_repair_ongoing
from backend.control_room import send_control_room_alert, buzzer_alert
from backend.train_control import send_train_stop_command

import random
import datetime

def simulate_sensor_health(sensor_name):
    now = datetime.datetime.now().strftime("%H:%M:%S")

    components = [
        "Vibration Module",
        "Sound Module",
        "CCTV Feed",
        "Drone Link",
        "Network Sync"
    ]

    health = {}

    for comp in components:
        status = random.choices(
            ["online", "degraded", "offline"],
            weights=[0.7, 0.2, 0.1]
        )[0]

        health[comp] = {
            "status": status,
            "last_update": now
        }

    return health

# =========================
# THEME STATE
# =========================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# =========================
# INCIDENT ACK STATE
# =========================
if "alert_acknowledged" not in st.session_state:

    if "operator_note" not in st.session_state:
        st.session_state.operator_note = ""

# 🔒 SAFETY RESET (IMPORTANT)
if "alert_acknowledged" not in st.session_state or isinstance(st.session_state.alert_acknowledged, bool):
    st.session_state.alert_acknowledged = {}

if "operator_note" not in st.session_state or isinstance(st.session_state.operator_note, str):
    st.session_state.operator_note = {}

if "incident_logs" not in st.session_state or isinstance(st.session_state.incident_logs, list):
    st.session_state.incident_logs = {}


# =====================================================
# HERO HEADER (DASHBOARD TITLE)
# =====================================================
# Ye top banner app ko professional
# control-room dashboard look deta hai
st.markdown("""
<div style="
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 25px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
">
    <h1 style="margin-bottom: 5px;">🚆 AI Railway Safety Control Room</h1>
    <p style="margin-top: 0;">
        Multisensor Monitoring • Real‑Time Decisions • Predictive Alerts
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# LOGOUT BUTTON
# =========================
with st.sidebar:
    st.markdown("### 👤 Operator Panel")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

# =====================================================
# SIDEBAR : CONTROL PANEL
# =====================================================
# Ye sidebar ek railway control room operator
# ke input panel jaisa kaam karta hai

import os
import streamlit as st

# Evidence folder ensure
os.makedirs("evidence", exist_ok=True)

# Browser audio permission note
st.sidebar.info("🔊 Audio alerts enabled after any interaction")

with st.sidebar:
    st.markdown("## 🚦 Control Panel")

    # =========================
    # 📸 CCTV UPLOAD
    # =========================
    uploaded_cctv = st.file_uploader(
        "📸 Upload CCTV Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_cctv is not None:
        with open("evidence/cctv.jpg", "wb") as f:
            f.write(uploaded_cctv.getbuffer())
        st.success("CCTV image uploaded")

    # =========================
    # 🚁 DRONE UPLOAD
    # =========================
    uploaded_drone = st.file_uploader(
        "🚁 Upload Drone Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_drone is not None:
        with open("evidence/drone.jpg", "wb") as f:
            f.write(uploaded_drone.getbuffer())
        st.success("Drone image uploaded")

    # =========================
    # 🔊 SOUND UPLOAD
    # =========================
    uploaded_sound = st.file_uploader(
        "🔊 Upload Sound Clip",
        type=["wav", "mp3"]
    )

    if uploaded_sound is not None:
        with open("evidence/sound.wav", "wb") as f:
            f.write(uploaded_sound.getbuffer())
        st.success("Sound clip uploaded")

    st.markdown("---")
    st.caption("📡 Sensor & Feed Inputs")

    # =========================
    # 📍 SENSOR LOCATION
    # =========================
    active_zone = st.session_state.active_zone

    zone_sensor_data = ZONE_SENSORS[active_zone]

    selected_sensor = st.selectbox(
        "📍 Select Sensor",
        list(zone_sensor_data.keys())
    )
    sensor_health = simulate_sensor_health(selected_sensor)


    sensor_details = zone_sensor_data[selected_sensor]

    sensor_name = sensor_details["name"]
    lat = sensor_details["lat"]
    lon = sensor_details["lon"]

    st.caption(f"📌 Location: {sensor_name},{lat}, {lon}")
    st.markdown(f"[🌍 View on Map](https://www.google.com/maps?q={lat},{lon})")
    st.markdown("---")
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    # =========================
    # 📈 VIBRATION CSV
    # =========================
    vibration_file = st.file_uploader(
        "📈 Upload Vibration CSV",
        type=["csv"]
    )

    if vibration_file is not None:
        st.success("Vibration data uploaded")

# =====================================================
# SENSOR ANALYSIS
# =====================================================
vibration_df = None
vibration_status = "no_data"

# CSV safely read kar rahe hain
if vibration_file is not None:
    try:
        vibration_df = pd.read_csv(vibration_file)

        # Column validation
        if "acceleration" not in vibration_df.columns:
            st.error("❌ CSV must contain an 'acceleration' column")
            vibration_df = None
            vibration_status = "invalid_format"
        else:
            vibration_status = analyze_vibration(vibration_df)

    except EmptyDataError:
        st.error("❌ Uploaded vibration CSV is empty")
        vibration_status = "empty_file"

    except Exception as e:
        st.error(f"❌ Error reading vibration CSV: {e}")
        vibration_status = "read_error"

# Other sensor analysis
cctv_result = analyze_visual(uploaded_cctv)
drone_result = analyze_visual(uploaded_drone)
sound_status = analyze_sound(uploaded_sound)
weather_status = fetch_weather_forecast(sensor_details["name"])

# Train related data
train = get_train_status()
direction = get_direction(train["distance_km"])

# Maintenance check
repair_ongoing = is_repair_ongoing(selected_sensor)


# =====================================================
# HELPER FUNCTION (SMART STATUS DISPLAY)
# =====================================================
def display_status(label, value):
    # Case 1: No data / no feed → very subtle
    if value in ["no_data", "no_feed"]:
        st.markdown(
            f"<div style='font-size:18px; color:#999;'>"
            f"{label}: {value.replace('_', ' ').title()}"
            f"</div>",
            unsafe_allow_html=True
        )

    # Case 2: Normal / low risk → small but visible
    elif value in ["normal", "low_risk"]:
        st.markdown(
            f"<div style='font-size:18px; color:green;'>"
            f"{label}: {value.replace('_', ' ').title()}"
            f"</div>",
            unsafe_allow_html=True
        )

    # Case 3: Anything important → big metric card
    else:
        st.markdown(
            f"<div style='font-size:24px; color:red;'>"
            f"{label}: {value.replace('_', ' ').title()}"
            f"</div>",
            unsafe_allow_html=True)
        
# =====================================================
# HELPER FUNCTION (RELIABLE AUDIO)
# =====================================================
def play_sound(sound_file):
    # Streamlit ka native audio (browser-safe)
    st.audio(sound_file)

# =========================
# DASHBOARD TABS
# =========================
tab_dashboard, tab_history, tab_health = st.tabs([
    "🚨 Live Dashboard",
    "📜 Incident History",
    "🏥 System Health"
])

with tab_dashboard:

    # =====================================================
    # ANALYSIS RESULTS (DASHBOARD CARDS)
    # =====================================================
    # Har sensor ka status card format me
    # dikhaya ja raha hai for quick understanding
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🔍 Analysis Results")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        display_status("📈 Vibration", vibration_status)


    with col2:
        display_status("📷 CCTV", cctv_result)

    with col3:
        display_status("🚁 Drone", drone_result)
    with col4:
        display_status("🎧 Sound", sound_status)

    with col5:
        display_status("🌦 Weather", weather_status)
    st.markdown("</div>", unsafe_allow_html=True)
    # Vibration graph alag se dikhate hain
    if vibration_df is not None:
        st.markdown("### 📊 Vibration Signal")
        st.line_chart(vibration_df["acceleration"])



    # =====================================================
    # TRAIN STATUS
    # =====================================================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🚆 Train Status")

    st.write("Train ID:", train["train_id"])
    st.write("Distance from Track (km):", train["distance_km"])
    st.write("Speed (km/h):", train["speed_kmph"])
    st.write("Direction:", direction)

    st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # DECISION ENGINE (FINAL)
    # =========================

    tampering_signals = []

    if vibration_status == "abnormal":
        tampering_signals.append("vibration")

    if sound_status == "suspicious":
        tampering_signals.append("sound")

    if cctv_result == "tampering":
        tampering_signals.append("cctv")

    if drone_result == "tampering":
        tampering_signals.append("drone")


    tampering_detected = len(tampering_signals) > 0

    final_status = "SAFE"
    final_action = "No action required"

    # Maintenance override
    if tampering_detected and repair_ongoing:
        final_status = "MAINTENANCE MODE"
        final_action = "Repair ongoing. Alerts suppressed."

    elif tampering_detected:

        # Control room always informed
        send_control_room_alert(
            f"Risk detected at {selected_sensor} | Signals: {', '.join(tampering_signals)}"
        )

        buzzer_alert()

        # Emergency only if train is close & approaching
        if direction == "approaching" and train["distance_km"] <= 7:
            final_status = "🚨 EMERGENCY"
            final_action = "STOP TRAIN IMMEDIATELY"

            send_train_stop_command(
                train["train_id"],
                lat,
                lon
            )

            send_control_room_alert(
                f"EMERGENCY STOP issued for Train {train['train_id']} "
                f"at ({lat}, {lon})"
            )

        else:
            final_status = "⚠️ WARNING"
            final_action = "Potential risk detected. Train not in immediate danger."

    else:
        final_status = "SAFE"
        final_action = "Track conditions normal."

    # =========================
    # LIVE ALERTS PANEL
    # =========================
    st.markdown("## 🚨 Live Alerts")

    if final_status.startswith("🚨"):
        st.markdown(f"""
        <div class="alert-card alert-emergency">
            🚨 EMERGENCY DETECTED <br>
            Sensor: {selected_sensor} <br>
            Action: {final_action}
        </div>
        """, unsafe_allow_html=True)

    elif final_status.startswith("⚠️"):
        st.markdown(f"""
        <div class="alert-card alert-warning">
            ⚠️ WARNING <br>
            Sensor: {selected_sensor} <br>
            Action: {final_action}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="alert-card alert-safe">
            ✅ ALL TRACKS SAFE <br>
            No active alerts right now
        </div>
        """, unsafe_allow_html=True)
    # =====================================================
    # 🧾 EVIDENCE PANEL (NON-BLOCKING)
    # =====================================================
    with st.container():
        st.markdown("## 🧾 Evidence Panel")

        if final_status.startswith(("🚨", "⚠️")):

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown("### 📸 CCTV")
                if os.path.exists("evidence/cctv.jpg"):
                    st.image("evidence/cctv.jpg", width=250)
                else:
                    st.info("No CCTV evidence")

            with col2:
                st.markdown("### 🚁 Drone")
                if os.path.exists("evidence/drone.jpg"):
                    st.image("evidence/drone.jpg", width=250)
                else:
                    st.info("No Drone evidence")

            with col3:
                st.markdown("### 🔊 Sound")
                if os.path.exists("evidence/sound.wav"):
                    st.audio("evidence/sound.wav")
                else:
                    st.info("No Sound evidence")

        else:
            st.info("System is SAFE — no evidence required")

    st.markdown("---")  # 👈 VERY IMPORTANT separator

    # =====================================================
    # OPERATOR ACTION PANEL
    # =====================================================
    # Ye section operator ko allow karta hai
    # ki wo alerts ko acknowledge kare aur
    # apne actions ko log kare.
        
    st.markdown("## 🧑‍✈️ Operator Action Panel")

    # Ensure zone keys exist
    st.session_state.alert_acknowledged.setdefault(active_zone, False)
    st.session_state.operator_note.setdefault(active_zone, "")
    st.session_state.incident_logs.setdefault(active_zone, [])

    if final_status.startswith(("🚨", "⚠️")):

        if not st.session_state.alert_acknowledged[active_zone]:
            st.warning("⚠️ Alert requires operator acknowledgement")

            note = st.text_area(
                "📝 Operator Note (optional)",
                placeholder="e.g. Informed maintenance team, monitoring situation...",
                key=f"note_{active_zone}"
            )

            if st.button("✅ Acknowledge Alert", key=f"ack_{active_zone}"):

                st.session_state.alert_acknowledged[active_zone] = True
                st.session_state.operator_note[active_zone] = note

                st.session_state.incident_logs[active_zone].append({
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "zone": active_zone,
                    "sensor": selected_sensor,
                    "location": sensor_name,
                    "status": final_status,
                    "action": final_action,
                    "note": note if note else "No note added"
                })

                st.success("Alert acknowledged successfully ✔️")

        else:
            st.success("✅ Alert already acknowledged")

            if st.session_state.operator_note[active_zone]:
                st.info(f"📝 Operator Note: {st.session_state.operator_note[active_zone]}")

    else:
        st.info("No active alerts to acknowledge 👍")

    # =====================================================
    # FINAL SYSTEM STATUS (CONTROL ROOM ALERTS)
    # =====================================================
    # Ye section operator ko clearly batata hai
    # ki system ka final decision kya hai

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🚦 System Status")

        # EMERGENCY STATE
    if final_status.startswith("🚨"):
            play_sound("assets/emergency_beep.mp3")
            st.markdown("""
            <div style="
                background-color:#ff4d4d;
                color:white;
                padding:20px;
                border-radius:10px;
                font-size:20px;
                font-weight:bold;
            ">
                🚨 EMERGENCY ALERT<br>
                Train Stop Command Issued Immediately
            </div>
            """, unsafe_allow_html=True)

        # WARNING STATE
    elif final_status.startswith("⚠️"):
            play_sound("assets/warning_beep.mp3")    
            st.markdown("""
            <div style="
                background-color:#ffa500;
                color:black;
                padding:20px;
                border-radius:10px;
                font-size:18px;
                font-weight:bold;
            ">
                ⚠️ WARNING<br>
                Control Room Notified
            </div>
            """, unsafe_allow_html=True)

        # MAINTENANCE MODE
    elif final_status == "MAINTENANCE MODE":
            st.markdown("""
            <div style="
                background-color:#ffd966;
                color:black;
                padding:20px;
                border-radius:10px;
                font-size:18px;
                font-weight:bold;
            ">
                🛠 MAINTENANCE MODE<br>
                Alerts Suppressed (Repair Ongoing)
            </div>
            """, unsafe_allow_html=True)

        # SAFE STATE
    else:
            st.markdown("""
            <div style="
                background-color:#4CAF50;
                color:white;
                padding:20px;
                border-radius:10px;
                font-size:18px;
                font-weight:bold;
            ">
                ✅ TRACK SAFE<br>
                No Action Required
            </div>
            """, unsafe_allow_html=True)

        # Extra info (operator clarity)
    st.write("📢 Final Action:", final_action)
    st.write("📍 Location (Lat, Lon):", f"{lat}, {lon}")

    map_iframe = f"""
    <iframe
        width="250"
        height="250"
        frameborder="0"
        style="border:0; border-radius:8px;"
        src="https://www.google.com/maps?q={lat},{lon}&z=15&output=embed"
        allowfullscreen>
    </iframe>
"""

    st.components.v1.html(map_iframe, height=250)

    with st.expander("🔍 Expand Map (Full Screen View)"):
        big_map_iframe = f"""
        <iframe
            width="100%"
            height="500"
            frameborder="0"
            style="border:0; border-radius:10px;"
            src="https://www.google.com/maps?q={lat},{lon}&z=17&output=embed"
            allowfullscreen>
        </iframe>
        """
        st.components.v1.html(big_map_iframe, height=520)


    st.markdown("</div>", unsafe_allow_html=True)


with tab_history:
    st.markdown("## 📜 Incident History")

    # 🔖 Zone label (top-right)
    st.markdown(
        f"<div style='text-align:right; font-weight:600;'>📍 Zone: {active_zone}</div>",
        unsafe_allow_html=True
    )

    zone_logs = st.session_state.incident_logs.get(active_zone, [])

    if len(zone_logs) > 0:

        table_data = []

        for incident in reversed(zone_logs):
            table_data.append({
                "Time": incident["time"],
                "Zone": incident["zone"],
                "Sensor": incident["sensor"],
                "Location": incident["location"],
                "Status": incident["status"],
                "Action Taken": incident["action"],
                "Operator Note": incident["note"]
            })

        import pandas as pd
        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            use_container_width=True,
            height=350
        )

    else:
        st.info("No incidents logged for this zone 👍")


with tab_health:

    st.markdown("## 🩺 System Health Status")
    st.markdown(f"📍 **Zone:** {active_zone}")
    st.markdown(f"🛰️ **Sensor:** {selected_sensor}")

    sensor_health = simulate_sensor_health(selected_sensor)

    for component, info in sensor_health.items():

        if info["status"] == "online":
            st.success(
                f"🟢 {component} ONLINE | Last update: {info['last_update']}"
            )

        elif info["status"] == "degraded":
            st.warning(
                f"🟡 {component} DEGRADED | Last update: {info['last_update']}"
            )

        else:
            st.error(
                f"🔴 {component} OFFLINE | Last update: {info['last_update']}"
            )
# Layout finalized
