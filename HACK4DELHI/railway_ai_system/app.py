import os
import sys
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
import datetime
# =========================
# PATH SETUP (FIXED)
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(APP_DIR, "images")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# =========================
# SAFE IMAGE LOADER
# =========================
def show_image(filename, **kwargs):
    img_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(img_path):
        st.image(img_path, **kwargs)
    else:
        st.warning(f"⚠️ Image not found: {filename}")



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




st.set_page_config(
    page_title="AI Railway Safety System",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)




def load_css():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")

    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ style.css not found, running without custom CSS")

load_css()

if "incident_logs" not in st.session_state:
    st.session_state.incident_logs = {}

if "alert_acknowledged" not in st.session_state:
    st.session_state.alert_acknowledged = {}

if "operator_note" not in st.session_state:
    st.session_state.operator_note = {}

if "current_alert_context" not in st.session_state:
    st.session_state.current_alert_context = {}

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

if "vibration_ml_details" not in st.session_state:
    st.session_state.vibration_ml_details = {}


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "active_zone" not in st.session_state:
    st.session_state.active_zone = None    

if not st.session_state.logged_in:
    st.markdown("""
        <div style="
            background-color:#f3f4f6;
            color:#000000;
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
    ["HOME", "RUN ANALYSIS", "ABOUT US", "CONTACT US"]
)
    
    with home_tab:
        show_image("banner.png", use_container_width=True)

        st.markdown("""
        <style>
        img {
            max-height: 260px;
            object-fit: cover;
        }
        </style>
        """, unsafe_allow_html=True)
            

        st.markdown("""
        # AI‑Enabled Railway Tampering Detection & Safety Monitoring Portal  
        ### Government of India | Smart Railway Infrastructure Initiative
        """)

        st.markdown("---")

        st.markdown("""
        
        Indian Railways operates a vast and geographically distributed railway network that is continuously exposed to **tampering‑related threats**, including **track interference, vandalism, sabotage, theft of railway assets, and negligence‑induced damage**.

        These activities, whether intentional or unintentional, pose **serious risks to passenger safety, train operations, and national infrastructure security**. Existing monitoring mechanisms rely heavily on **manual inspections and isolated surveillance systems**, resulting in delayed detection and reactive response.

        The absence of an **integrated, real‑time, tampering‑focused monitoring platform** limits the ability of authorities to proactively identify threats, correlate sensor data with evidence, and ensure structured accountability.
        """)

        st.markdown("---")

        st.markdown("""
        ## Need for a Government‑Grade Solution

        There is a critical requirement for a **secure, centralized, and governance‑ready digital platform** capable of detecting tampering‑related anomalies in real time, supporting **human‑in‑the‑loop decision making**, and maintaining **zone‑wise operational independence**.

        Such a system must be specifically designed for **government operations**, prioritizing reliability, transparency, auditability, and controlled response mechanisms.
        """)
        col_img, col_text = st.columns([1, 2])

        with col_img:
            show_image("collage.png", use_container_width=True)
            
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

        col_text, col_img = st.columns([2, 1])

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
           show_image("controlroom.png", use_container_width=True)


        st.markdown("---")

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

        st.markdown("""
        ## Vision

        The portal is envisioned as a **national‑grade digital safety backbone** for railway operations, enabling authorities to transition from **reactive incident handling to proactive tampering prevention**, supported by **AI‑assisted intelligence and structured governance frameworks**.
        """)

        st.markdown("")

        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            show_image("delhi.png", use_container_width=True)
        st.markdown("---")

        st.success(
            "This portal is designed for authorized government personnel and supports secure, zone‑isolated railway safety operations."
        )


    with run_tab:
        st.title("Railway Safety System – Login")

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
        st.markdown("## About Us")

        st.write("""
        **AI Railway Safety System** is a smart monitoring platform
        developed to enhance railway track safety using
        multi‑sensor intelligence and automated decision making.
        """)

        st.markdown("### Our Objective")
        st.markdown("""
        - Prevent railway accidents  
        - Detect track tampering in real‑time  
        - Assist control rooms with fast decisions  
        """)

        st.markdown("### Technology Stack")
        st.markdown("""
        - Machine Learning & Signal Processing  
        - Computer Vision (CCTV & Drone feeds)  
        - Acoustic & Vibration Analysis  
        - Real‑time Decision Engine  
        """)
    with contact_tab:
        st.markdown("## Contact Us")

        st.write("""
        For system access, collaboration, or technical queries,
        please reach out to the development team.
        """)

        st.markdown("### Email")
        st.write("railwaysafety.ai@gmail.com")

        st.markdown("### Organization")
        st.write("Railway Safety Innovation Lab")

        st.markdown("### Support")
        st.write("Available during demo and evaluation sessions.")

    st.stop()



from railway_ai_system.backend.cctv import analyze_visual
from railway_ai_system.backend.vibration import analyze_vibration
from railway_ai_system.backend.drone import analyze_drone_image
from railway_ai_system.backend.sound import analyze_sound
from railway_ai_system.backend.train_schedule import get_train_status, get_direction
from railway_ai_system.backend.maintenance_db import is_repair_ongoing
from railway_ai_system.backend.control_room import send_control_room_alert, buzzer_alert
from railway_ai_system.backend.train_control import send_train_stop_command
from railway_ai_system.backend.weather import get_zone_weather, fetch_weather_by_coordinates


import random
import datetime
# =========================
# 🚨 HARD CCTV MODEL STATUS WARNING
# =========================

if not is_model_loaded():
    st.error(
        "🚨 **CCTV AI MODEL NOT LOADED**\n\n"
        "- CCTV predictions are running in **FALLBACK MODE**\n"
        "- All outputs will default to **NORMAL**\n"
        "- 🚫 NO AI inference is happening\n\n"
        "**Fix Required:**\n"
        "- Ensure `my_image_classifier.pkl` exists\n"
        "- Ensure correct model path\n"
        "- Check deployment logs\n",
        icon="🚨"
    )
else:
    st.success("✅ CCTV AI model loaded successfully", icon="🤖")



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

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


st.markdown("""
<div style="
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 25px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
">
    <h1 style="margin-bottom: 5px;"> AI Railway Safety Control Room</h1>
    <p style="margin-top: 0;">
        Multisensor Monitoring • Real‑Time Decisions • Predictive Alerts
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 👤 Operator Panel")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()


import os
import streamlit as st

os.makedirs("evidence", exist_ok=True)

st.sidebar.info("🔊 Audio alerts enabled after any interaction")

with st.sidebar:
    st.markdown("## 🚦 Control Panel")

    uploader_key = st.session_state.file_uploader_key

    uploaded_cctv = st.file_uploader(
        "Upload CCTV Image",
        type=["jpg", "jpeg", "png"],
        key=f"cctv_{uploader_key}"
    )

    if uploaded_cctv is not None:
        with open("evidence/cctv.jpg", "wb") as f:
            f.write(uploaded_cctv.getbuffer())
        st.success("CCTV image uploaded")

    uploaded_drone = st.file_uploader(
        "Upload Drone Image",
        type=["jpg", "jpeg", "png"],
        key=f"drone_{uploader_key}"
    )

    if uploaded_drone is not None:
        with open("evidence/drone.jpg", "wb") as f:
            f.write(uploaded_drone.getbuffer())
        st.success("Drone image uploaded")

    uploaded_sound = st.file_uploader(
        "Upload Sound Clip",
        type=["wav", "mp3"],
        key=f"sound_{uploader_key}"
    )

    if uploaded_sound is not None:
        with open("evidence/sound.wav", "wb") as f:
            f.write(uploaded_sound.getbuffer())
        st.success("Sound clip uploaded")

    st.markdown("---")
    st.caption("Sensor & Feed Inputs")

    active_zone = st.session_state.active_zone

    zone_sensor_data = ZONE_SENSORS[active_zone]
    zone_weather = get_zone_weather(zone_sensor_data)


    selected_sensor = st.selectbox(
        "📍 Select Sensor",
        list(zone_sensor_data.keys())
    )
    sensor_health = simulate_sensor_health(selected_sensor)


    sensor_details = zone_sensor_data[selected_sensor]

    sensor_name = sensor_details["name"]
    lat = sensor_details["lat"]
    lon = sensor_details["lon"]
    from backend.weather import fetch_weather_by_coordinates
    weather = fetch_weather_by_coordinates(lat, lon)


    st.caption(f"📌 Location: {sensor_name},{lat}, {lon}")
    st.markdown(f"[View on Map](https://www.google.com/maps?q={lat},{lon})")
    st.markdown("---")
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"

    vibration_file = st.file_uploader(
        "Upload Vibration CSV",
        type=["csv"],
        key=f"vibration_{uploader_key}"
    )

    if vibration_file is not None:
        st.success("Vibration data uploaded")

vibration_df = None
vibration_status = "no_data"
vibration_details = None

if vibration_file is not None:
    try:
        vibration_df = pd.read_csv(vibration_file)

        if "acceleration" not in vibration_df.columns:
            st.error("CSV must contain an 'acceleration' column")
            vibration_df = None
            vibration_status = "invalid_format"
        else:
            # Call updated analyze_vibration that returns tuple
            result = analyze_vibration(vibration_df)
            
            # Check if result is tuple (ML enabled) or string (fallback)
            if isinstance(result, tuple):
                vibration_status, vibration_details = result
                st.session_state.vibration_ml_details[active_zone] = vibration_details
            else:
                vibration_status = result
                vibration_details = None
                st.session_state.vibration_ml_details[active_zone] = None

    except EmptyDataError:
        st.error("Uploaded vibration CSV is empty")
        vibration_status = "empty_file"

    except Exception as e:
        st.error(f"Error reading vibration CSV: {e}")
        vibration_status = "read_error"

cctv_result = analyze_visual(uploaded_cctv)
drone_result = analyze_drone_image(uploaded_drone)
sound_status = analyze_sound(uploaded_sound)

train = get_train_status()
direction = get_direction(train["distance_km"])

repair_ongoing = is_repair_ongoing(selected_sensor)


def display_status(label, value):
    if value in ["no_data", "no_feed"]:
        st.markdown(
            f"<div style='font-size:18px; color:#999;'>"
            f"{label}: {value.replace('_', ' ').title()}"
            f"</div>",
            unsafe_allow_html=True
        )

    elif value in ["normal", "low_risk"]:
        st.markdown(
            f"<div style='font-size:18px; color:green;'>"
            f"{label}: {value.replace('_', ' ').title()}"
            f"</div>",
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"<div style='font-size:24px; color:red;'>"
            f"{label}: {value.replace('_', ' ').title()}"
            f"</div>",
            unsafe_allow_html=True)
        
def play_sound(sound_file):
    st.audio(sound_file)

tab_dashboard, tab_history, tab_health = st.tabs([
    "Live Dashboard",
    "Incident History",
    "System Health"
])

with tab_dashboard:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Analysis Results")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        display_status("Vibration", vibration_status)
        
        # Display ML details if available
        if vibration_details and vibration_status in ["normal", "abnormal"]:
            with st.expander("🤖 ML Analysis Details"):
                st.markdown(f"**Model Type:** Logistic Regression")
                st.markdown(f"**Confidence:** {vibration_details['confidence']:.1f}%")
                st.markdown(f"**Probability:**")
                st.markdown(f"- Normal: {vibration_details['prob_normal']:.3f}")
                st.markdown(f"- Abnormal: {vibration_details['prob_abnormal']:.3f}")
                
                if vibration_status == "abnormal" and vibration_details.get('top_features'):
                    st.markdown("**Top Indicators:**")
                    for feat in vibration_details['top_features']:
                        st.markdown(f"- {feat['name']}: {feat['value']:.3f}")
                
                st.markdown("**Key Metrics:**")
                st.markdown(f"- RMS: {vibration_details['rms']:.3f}")
                st.markdown(f"- Peak: {vibration_details['peak']:.3f}")
                st.markdown(f"- Crest Factor: {vibration_details['crest_factor']:.2f}")
                st.markdown(f"- Zero Crossing Rate: {vibration_details['zero_crossing_rate']:.4f}")


    with col2:
        if cctv_result == "normal":
            st.success("Track Status: NORMAL")

        elif "human" in cctv_result:
            st.error("🚨 ALERT: Human detected on track")

        elif "object" in cctv_result:
            st.error("🚨 ALERT: Object detected on track")

        elif "severe" in cctv_result:
            st.error("🚨 ALERT: Severe Tampering Detected")

        elif cctv_result == "no_feed":
            st.warning("⚠️ No CCTV feed available")

        else:
            st.warning(f"⚠️ CCTV Result: {cctv_result}")

    with col3:
        if drone_result == "normal":
            st.success("✅ Drone Status: NORMAL")

        elif drone_result == "anomaly":
            st.error("🚨 ALERT: Anomaly detected by drone")

        elif drone_result == "no_feed":
            st.warning("⚠️ No drone feed available")

        else:
            st.warning(f"⚠️ Drone Result: {drone_result}")

        
    with col4:
        display_status("🎧 Sound", sound_status)

        
    st.markdown("#### 🌦 Live Weather")

    st.markdown(f"""
    ###### 📍 {sensor_details['name']}
    - Temperature: {weather['temperature_c']} °C
    - Humidity: {weather['humidity_percent']} %
    - Wind Speed: {weather['wind_speed_mps']} m/s
    - Rain (last 1h): {weather['rain_mm_last_1h']} mm
    - Condition: {weather['description']}
    """)


    st.markdown("</div>", unsafe_allow_html=True)

    if vibration_df is not None:
        st.markdown("### Vibration Signal")
        chart_data = vibration_df.set_index('time') if 'time' in vibration_df.columns else vibration_df
        
        st.line_chart(chart_data["acceleration"], 
                     x_label="Time (seconds)" if 'time' in vibration_df.columns else "Sample Number",
                     y_label="Acceleration (g)")



    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Train Status")

    st.write("Train ID:", train["train_id"])
    st.write("Distance from Track (km):", train["distance_km"])
    st.write("Speed (km/h):", train["speed_kmph"])
    st.write("Direction:", direction)

    st.markdown("</div>", unsafe_allow_html=True)


    tampering_signals = []

    if vibration_status == "abnormal":
        tampering_signals.append("vibration")

    if sound_status == "suspicious":
        tampering_signals.append("sound")

    if cctv_result == "suspicious(human detected)":
        tampering_signals.append("cctv")

    if cctv_result == "tampering(object detected)":
        tampering_signals.append("cctv")    

    if cctv_result == "tampering(broken track)":
        tampering_signals.append("cctv")

    if drone_result == "anomaly":
        tampering_signals.append("drone")


    tampering_detected = len(tampering_signals) > 0

    final_status = "SAFE"
    final_action = "No action required"

    if tampering_detected and repair_ongoing:
        final_status = "MAINTENANCE MODE"
        final_action = "Repair ongoing. Alerts suppressed."

    elif tampering_detected:

        send_control_room_alert(
            f"⚠️ Risk detected at {selected_sensor} | Signals: {', '.join(tampering_signals)}"
        )

        buzzer_alert()

        if direction == "approaching" and train["distance_km"] <= 7:
            final_status = "🚨 EMERGENCY"
            final_action = "STOP TRAIN IMMEDIATELY"

            send_train_stop_command(
                train["train_id"],
                lat,
                lon
            )

            send_control_room_alert(
                f"🚨 EMERGENCY STOP issued for Train {train['train_id']} "
                f"at ({lat}, {lon})"
            )

        else:
            final_status = "⚠️ WARNING"
            final_action = "Potential risk detected. Train not in immediate danger."

    else:
        final_status = "SAFE"
        final_action = "Track conditions normal."

    alert_context_id = f"{active_zone}_{selected_sensor}_{final_status}_{'-'.join(tampering_signals)}"

    if st.session_state.current_alert_context.get(active_zone) != alert_context_id:
        st.session_state.alert_acknowledged[active_zone] = False
        st.session_state.operator_note[active_zone] = ""
        st.session_state.current_alert_context[active_zone] = alert_context_id

    st.markdown("## Live Alerts")

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

    with st.container():
        st.markdown("## Evidence Panel")

        if final_status.startswith(("🚨", "⚠️")):

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown("### CCTV")
                if os.path.exists("evidence/cctv.jpg"):
                    st.image("evidence/cctv.jpg", width=250)
                else:
                    st.info("No CCTV evidence")

            with col2:
                st.markdown("### Drone")
                if os.path.exists("evidence/drone.jpg"):
                    st.image("evidence/drone.jpg", width=250)
                else:
                    st.info("No Drone evidence")

            with col3:
                st.markdown("### Sound")
                if os.path.exists("evidence/sound.wav"):
                    st.audio("evidence/sound.wav")
                else:
                    st.info("No Sound evidence")

        else:
            st.info("System is SAFE — no evidence required")

    st.markdown("---")

        
    st.markdown("## Operator Action Panel")

    st.session_state.alert_acknowledged.setdefault(active_zone, False)
    st.session_state.operator_note.setdefault(active_zone, "")
    st.session_state.incident_logs.setdefault(active_zone, [])

    if final_status.startswith(("🚨", "⚠️")):

        if not st.session_state.alert_acknowledged[active_zone]:
            st.warning("⚠️ Alert requires operator acknowledgement")

            note = st.text_area(
                "Operator Note (optional)",
                placeholder="e.g. Informed maintenance team, monitoring situation...",
                key=f"note_{active_zone}_{alert_context_id}"
            )

            if st.button("✅ Acknowledge Alert", key=f"ack_{active_zone}_{alert_context_id}"):

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

                evidence_files = [
                    "evidence/cctv.jpg",
                    "evidence/drone.jpg",
                    "evidence/sound.wav"
                ]
                
                for file_path in evidence_files:
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            st.warning(f"Could not remove {file_path}: {e}")

                st.session_state.file_uploader_key += 1

                st.success("✔️ Alert acknowledged and logged to incident history")
                st.info("📁 Evidence files cleared - ready for next analysis")
                
                st.rerun()

        else:
            st.success("✅ Alert acknowledged for current incident")

            if st.session_state.operator_note[active_zone]:
                st.info(f"Operator Note: {st.session_state.operator_note[active_zone]}")
            
            st.caption("System will reset when sensor/status changes")

    else:
        st.info("No active alerts to acknowledge 👍")


    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("System Status")

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
                MAINTENANCE MODE<br>
                Alerts Suppressed (Repair Ongoing)
            </div>
            """, unsafe_allow_html=True)

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

    st.write("Final Action:", final_action)
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

    with st.expander("Expand Map (Full Screen View)"):
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
    st.markdown("## Incident History")

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

    st.markdown("## System Health Status")
    st.markdown(f"📍 **Zone:** {active_zone}")
    st.markdown(f"**Sensor:** {selected_sensor}")

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














