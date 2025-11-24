# streamlit_app.py
"""
Clean Streamlit CCTV app (no hero banner).
- Sidebar (when logged out): Home, Register, Login, Settings (public)
- Sidebar (when logged in): Monitor, Settings, Logout
- Public Settings page available before login (persistent to settings.json)
- Register -> OTP -> Monitor
- Login -> Monitor
- Alerts: crowd threshold (immediate) OR weapon/violence continuous durations
- Safe DB migrations, st.query_params avoided for GetStart (removed), st.rerun() used
- Uses uploaded file path (HERO_IMG) if needed elsewhere
"""

import os
import time
import json
import random
import string
import sqlite3
import bcrypt
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from email.message import EmailMessage
import smtplib
from email_validator import validate_email, EmailNotValidError
from pathlib import Path

# -------------------------
# CONFIG / ENV
# -------------------------
# Email credentials
SMTP_EMAIL = st.secrets["SMTP_EMAIL"]
SMTP_APP_PASSWORD = st.secrets["SMTP_APP_PASSWORD"]

# Telegram credentials
TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

# Model paths
CROWD_MODEL_PATH = st.secrets.get("CROWD_MODEL_PATH", "People_count_model/best.pt")
WEAPON_MODEL_PATH = st.secrets.get("WEAPON_MODEL_PATH", "weapon_detection_model/best1.pt")

DB_PATH = "users.db"
SETTINGS_PATH = "settings.json"
SESSION_TOKEN_FILE = ".session_token"

# Uploaded/asset image path from conversation (kept available)
HERO_IMG = "assets/ai_mon.jpg"

# Default settings
DEFAULT_SETTINGS = {
    "crowd_threshold": 1000,
    "weapon_duration": 3.0,
    "violence_duration": 3.0,
    "alert_cooldown": 20
}

# load/save settings
def load_settings():
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r") as f:
                s = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in s:
                    s[k] = v
            return s
        except Exception:
            return DEFAULT_SETTINGS.copy()
    else:
        return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)

settings = load_settings()

# -------------------------
# DATABASE (sqlite) with safe migrations
# -------------------------
con = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = con.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password_hash BLOB,
    verified INTEGER DEFAULT 0,
    remember_token TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS otps (
    email TEXT,
    otp TEXT,
    created_at INTEGER
)
""")

# minimal alerts table, we'll migrate columns safely
cur.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT
)
""")
con.commit()

def safe_add_column(table, column_def):
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
        con.commit()
    except sqlite3.OperationalError:
        pass

safe_add_column("alerts", "type TEXT")
safe_add_column("alerts", "details TEXT")
safe_add_column("alerts", "created_at INTEGER")
safe_add_column("alerts", "snapshot_path TEXT")

# -------------------------
# UTILITIES: email, telegram, otp, alerts
# -------------------------
def send_email_html(to_email, subject, html_body, text_body=None):
    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email
        if text_body:
            msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
            smtp.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print("Email error:", e)
        return False

def send_telegram_alert(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        return r.status_code in (200,201)
    except Exception as e:
        print("Telegram error:", e)
        return False

def gen_otp(n=6):
    return "".join(random.choices(string.digits, k=n))

def save_otp(email, otp):
    cur.execute("INSERT INTO otps (email, otp, created_at) VALUES (?, ?, ?)", (email, otp, int(time.time())))
    con.commit()

def verify_otp(email, otp, expiry=300):
    cur.execute("SELECT otp, created_at FROM otps WHERE email=? ORDER BY created_at DESC LIMIT 1", (email,))
    r = cur.fetchone()
    if not r:
        return False
    saved, created = r
    if int(time.time()) - created > expiry:
        return False
    return saved == otp

def log_alert(alert_type, details, snapshot_path=None):
    cur.execute("INSERT INTO alerts (type, details, created_at, snapshot_path) VALUES (?, ?, ?, ?)",
                (alert_type, details, int(time.time()), snapshot_path))
    con.commit()

# -------------------------
# AUTH helpers
# -------------------------
def register_user(email, password):
    try:
        v = validate_email(email)
        email = v["email"]
    except EmailNotValidError as e:
        return False, str(e)
    pwd_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        cur.execute("INSERT OR REPLACE INTO users (email, password_hash, verified) VALUES (?, ?, 0)", (email, pwd_hash))
        con.commit()
        return True, "Registered"
    except Exception as e:
        return False, f"DB Error: {e}"

def mark_verified(email):
    cur.execute("UPDATE users SET verified=1 WHERE email=?", (email,))
    con.commit()

def check_login(email, password):
    cur.execute("SELECT password_hash, verified FROM users WHERE email=?", (email,))
    r = cur.fetchone()
    if not r:
        return False, "No account"
    pwd_hash, verified = r
    if not bcrypt.checkpw(password.encode(), pwd_hash):
        return False, "Wrong password"
    if verified == 0:
        return False, "Not verified"
    return True, "OK"

def set_remember_token(email, token):
    cur.execute("UPDATE users SET remember_token=? WHERE email=?", (token, email))
    con.commit()

def get_user_by_token(token):
    cur.execute("SELECT email FROM users WHERE remember_token=?", (token,))
    r = cur.fetchone()
    return r[0] if r else None

# -------------------------
# Load models (optional)
# -------------------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

@st.cache_resource
def load_models():
    if YOLO is None:
        return None, None
    crowd = None
    weapon = None
    try:
        crowd = YOLO(CROWD_MODEL_PATH)
    except Exception as e:
        print("Crowd model load error:", e)
    try:
        weapon = YOLO(WEAPON_MODEL_PATH)
    except Exception as e:
        print("Weapon model load error:", e)
    return crowd, weapon

crowd_model, weapon_model = load_models()

# -------------------------
# Detector transformer (webrtc)
# -------------------------
class DetectorTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_alert_time = 0
        self.first_seen = {"weapon": None, "violence": None}
        self.alerted = {"weapon": False, "violence": False}

    def _names_boxes(self, results):
        out = []
        try:
            if not results or len(results) == 0:
                return out
            r = results[0]
            if not hasattr(r, "boxes"):
                return out
            boxes = r.boxes
            cls_arr = boxes.cls.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            xy_arr = boxes.xyxy.cpu().numpy()
            model_names = getattr(weapon_model, "names", None)
            for i, cid in enumerate(cls_arr):
                cid = int(cid)
                name = model_names[cid] if model_names and cid in model_names else str(cid)
                conf = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                xy = xy_arr[i] if i < len(xy_arr) else None
                out.append((name, conf, xy))
        except Exception as e:
            print("names_boxes error:", e)
        return out

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        vis = img.copy()
        now = time.time()

        crowd_count = 0
        weapon_detected = False
        violence_detected = False
        detected_weapon_labels = []

        # CROWD (people) - GREEN boxes
        if crowd_model is not None:
            try:
                r = crowd_model(img, verbose=False)
                if len(r) > 0 and hasattr(r[0], "boxes") and hasattr(r[0].boxes, "xyxy"):
                    boxes = r[0].boxes.xyxy.cpu().numpy()
                    crowd_count = len(boxes)
                    for b in boxes:
                        x1, y1, x2, y2 = map(int, b[:4])
                        cv2.rectangle(vis, (x1,y1),(x2,y2), (0,255,0), 2)
            except Exception as e:
                print("crowd error:", e)

        # WEAPON / VIOLENCE (RED boxes)
        if weapon_model is not None:
            try:
                res = weapon_model(img, conf=0.45)
                items = self._names_boxes(res)
                for name, conf, xy in items:
                    if xy is None:
                        continue
                    x1, y1, x2, y2 = map(int, xy[:4])
                    lname = name.strip().lower()
                    if "violence" in lname:
                        violence_detected = True
                        cv2.rectangle(vis, (x1,y1),(x2,y2), (0,0,255), 3)
                        cv2.putText(vis, "Violence", (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    elif any(k in lname for k in ["gun","guns","knife","knives","weapon"]):
                        weapon_detected = True
                        detected_weapon_labels.append(name)
                        cv2.rectangle(vis, (x1,y1),(x2,y2), (0,0,255), 3)
                        cv2.putText(vis, "Weapon", (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    else:
                        cv2.rectangle(vis, (x1,y1),(x2,y2), (0,0,255), 1)
            except Exception as e:
                print("weapon error:", e)

        # overlay header
        h,w = vis.shape[:2]
        cv2.rectangle(vis, (0,0), (w,60), (0,0,0), -1)
        status_text = f"Crowd: {crowd_count}   |   Weapon: {'Yes' if weapon_detected else 'No'}   |   Violence: {'Yes' if violence_detected else 'No'}"
        cv2.putText(vis, status_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # ALERT LOGIC (settings-driven)
        crowd_threshold = settings.get("crowd_threshold", DEFAULT_SETTINGS["crowd_threshold"])
        weapon_dur = settings.get("weapon_duration", DEFAULT_SETTINGS["weapon_duration"])
        violence_dur = settings.get("violence_duration", DEFAULT_SETTINGS["violence_duration"])
        cooldown = settings.get("alert_cooldown", DEFAULT_SETTINGS["alert_cooldown"])

        # Crowd alert (immediate)
        if crowd_count > crowd_threshold and (now - getattr(self, "last_alert_time", 0) > cooldown):
            alert_msg = f"🚨 CROWD ALERT 🚨\nCrowd: {crowd_count}\nTime: {time.ctime()}"
            send_telegram_alert(alert_msg)
            cur.execute("SELECT email FROM users WHERE verified=1")
            rows = cur.fetchall()
            if rows:
                send_email_html(",".join([r[0] for r in rows]), "Crowd Alert", f"<pre>{alert_msg}</pre>", text_body=alert_msg)
            log_alert("crowd", alert_msg)
            self.last_alert_time = now

        # Weapon continuous detection
        if weapon_detected:
            if self.first_seen["weapon"] is None:
                self.first_seen["weapon"] = now
            if (now - self.first_seen["weapon"] >= float(weapon_dur)) and (not self.alerted["weapon"]) and (now - getattr(self, "last_alert_time", 0) > cooldown):
                alert_msg = f"🚨 WEAPON ALERT 🚨\nDetected: {', '.join(sorted(set(detected_weapon_labels)))}\nTime: {time.ctime()}"
                send_telegram_alert(alert_msg)
                cur.execute("SELECT email FROM users WHERE verified=1")
                rows = cur.fetchall()
                if rows:
                    send_email_html(",".join([r[0] for r in rows]), "Weapon Alert", f"<pre>{alert_msg}</pre>", text_body=alert_msg)
                log_alert("weapon", alert_msg)
                self.alerted["weapon"] = True
                self.last_alert_time = now
        else:
            self.first_seen["weapon"] = None
            self.alerted["weapon"] = False

        # Violence continuous detection
        if violence_detected:
            if self.first_seen["violence"] is None:
                self.first_seen["violence"] = now
            if (now - self.first_seen["violence"] >= float(violence_dur)) and (not self.alerted["violence"]) and (now - getattr(self, "last_alert_time", 0) > cooldown):
                alert_msg = f"🚨 VIOLENCE ALERT 🚨\nTime: {time.ctime()}"
                send_telegram_alert(alert_msg)
                cur.execute("SELECT email FROM users WHERE verified=1")
                rows = cur.fetchall()
                if rows:
                    send_email_html(",".join([r[0] for r in rows]), "Violence Alert", f"<pre>{alert_msg}</pre>", text_body=alert_msg)
                log_alert("violence", alert_msg)
                self.alerted["violence"] = True
                self.last_alert_time = now
        else:
            self.first_seen["violence"] = None
            self.alerted["violence"] = False

        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

# -------------------------
# UI (landing / auth / monitor / settings)
# -------------------------
st.set_page_config(page_title="Security CCTV", layout="wide", initial_sidebar_state="collapsed")

# Simple CSS
st.markdown("""
<style>
body { background: linear-gradient(180deg,#030615,#071027); color: #E6EEF8; }
.card { background: #0f1724; padding:18px; border-radius:10px; margin-bottom:18px; }
.small { color: #9aa7b8; }
</style>
""", unsafe_allow_html=True)

# session state keys
if "user" not in st.session_state:
    st.session_state.user = None
if "just_registered" not in st.session_state:
    st.session_state.just_registered = None
if "show_monitor" not in st.session_state:
    st.session_state.show_monitor = False
if "show_forgot" not in st.session_state:
    st.session_state.show_forgot = False

# Sidebar
with st.sidebar:
    st.markdown("<h4>Navigation</h4>", unsafe_allow_html=True)
    if not st.session_state.get("user"):
        nav = st.radio("", ["Home", "Register", "Login", "Settings"], index=0)
    else:
        nav = st.radio("", ["Monitor", "Logout"], index=0)
# GLOBAL LOGOUT HANDLER (MUST be right after sidebar)
if nav == "Logout":
    st.session_state.user = None
    st.session_state.show_monitor = False
    
    try:
        if os.path.exists(SESSION_TOKEN_FILE):
            os.remove(SESSION_TOKEN_FILE)
    except:
        pass

    st.rerun()


# ROUTING - logged out
if not st.session_state.get("user"):
    if nav == "Home":

        st.markdown(
                """
                <h1 style='text-align:center; margin-top:10px;'>
                    Real-Time Crowd Behavior & Anomaly Detection System
                </h1>
                """,
                unsafe_allow_html=True
            )


        col_left, col_right = st.columns([1.2, 1.2])

        # LEFT — clean text, no HTML
        with col_left:
            st.subheader("System Features")

            st.write(
                """
                  A complete AI-powered surveillance system that detects:

    - Crowd gatherings  
    - Weapon activity (Gun / Knife)  
    - Violence or aggressive activity  

    Alerts are automatically sent to your Gmail and Telegram.

    ### Key Features:
    - Crowd Counting (YOLO Model)  
    - Weapon Detection  
    - Violence Detection  
    - Real-time Alerts  
    - Live Webcam or Video Upload support  

    Use the sidebar to Register or Login and start monitoring.
    """
            )

        # RIGHT — image only
        with col_right:
            st.image("assets/ai_mon.jpg", width="stretch")





    # Register page
    if nav == "Register":
        st.markdown("<div class='card' style='max-width:800px; margin:20px auto;'>", unsafe_allow_html=True)
        st.subheader("Create account")
        r_email = st.text_input("Email", key="r_email")
        r_pwd = st.text_input("Password", type="password", key="r_pwd")
        if st.button("Register & Send OTP"):
            if not r_email or not r_pwd:
                st.error("Enter email and password")
            else:
                ok, msg = register_user(r_email, r_pwd)
                if not ok:
                    st.error(msg)
                else:
                    otp = gen_otp()
                    save_otp(r_email, otp)
                    sent = False
                    if SMTP_EMAIL and SMTP_APP_PASSWORD:
                        html = f"<h3>Your OTP</h3><p style='font-size:20px'><b>{otp}</b></p>"
                        sent = send_email_html(r_email, "Your OTP", html)
                    st.session_state.just_registered = r_email
                    st.success("Registered. Enter OTP below to verify.")
                    if not sent:
                        st.info(f"SMTP not configured — OTP (dev): {otp}")

        if st.session_state.get("just_registered"):
            st.markdown("---")
            st.subheader("Verify OTP")
            otp_in = st.text_input("OTP", key="verify_otp")
            col1, col2 = st.columns([2,1])
            with col2:
                if st.button("Verify OTP"):
                    if verify_otp(st.session_state.just_registered, otp_in):
                        mark_verified(st.session_state.just_registered)
                        st.success("Verified — redirecting to Monitor")
                        st.session_state.user = st.session_state.just_registered
                        st.session_state.just_registered = None
                        st.session_state.show_monitor = True
                        st.rerun()
                    else:
                        st.error("Invalid or expired OTP")
            with col1:
                if st.button("Resend OTP"):
                    otp2 = gen_otp()
                    save_otp(st.session_state.just_registered, otp2)
                    sent2 = False
                    if SMTP_EMAIL and SMTP_APP_PASSWORD:
                        sent2 = send_email_html(st.session_state.just_registered, "Your OTP", f"<b>{otp2}</b>")
                    if sent2:
                        st.success("OTP resent to your email")
                    else:
                        st.info(f"OTP (dev): {otp2}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Login page
    if nav == "Login":
        st.markdown("<div class='card' style='max-width:800px; margin:20px auto;'>", unsafe_allow_html=True)
        st.subheader("Login")
        l_email = st.text_input("Email", key="l_email")
        l_pwd = st.text_input("Password", type="password", key="l_pwd")
        remember = st.checkbox("Remember this device", key="l_remember")
        if st.button("Login"):
            ok, msg = check_login(l_email, l_pwd)
            if ok:
                st.session_state.user = l_email
                st.success("Logged in — redirecting to Monitor")
                if remember:
                    token = "".join(random.choices(string.ascii_letters+string.digits, k=64))
                    set_remember_token(l_email, token)
                    try:
                        with open(SESSION_TOKEN_FILE, "w") as f:
                            f.write(token)
                    except:
                        pass
                st.session_state.show_monitor = True
                st.rerun()
            else:
                st.error(msg)
        if st.button("Forgot Password"):
            st.session_state.show_forgot = True
        st.markdown("</div>", unsafe_allow_html=True)

    # Public Settings page (available before login)
    if nav == "Settings":
        st.markdown("<div class='card' style='max-width:900px; margin:20px auto;'>", unsafe_allow_html=True)
        st.subheader("Settings (public)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ct = st.number_input("Crowd threshold", min_value=1, value=int(settings.get("crowd_threshold",1000)), step=10, key="public_ct")
        with c2:
            wd = st.number_input("Weapon duration (s)", min_value=0.5, value=float(settings.get("weapon_duration",3.0)), step=0.5, key="public_wd")
        with c3:
            vd = st.number_input("Violence duration (s)", min_value=0.5, value=float(settings.get("violence_duration",3.0)), step=0.5, key="public_vd")
        with c4:
            ac = st.number_input("Alert cooldown (s)", min_value=1, value=int(settings.get("alert_cooldown",20)), step=1, key="public_ac")
        if st.button("Save settings (public)"):
            settings["crowd_threshold"] = int(ct)
            settings["weapon_duration"] = float(wd)
            settings["violence_duration"] = float(vd)
            settings["alert_cooldown"] = int(ac)
            save_settings(settings)
            st.success("Settings saved (applied globally).")
        st.markdown("</div>", unsafe_allow_html=True)

    # Forgot password flow
    if st.session_state.get("show_forgot"):
        st.markdown("<div class='card' style='max-width:800px; margin:20px auto;'>", unsafe_allow_html=True)
        st.subheader("Forgot Password")
        fp_email = st.text_input("Registered email", key="fp_email")
        if st.button("Send Reset OTP"):
            cur.execute("SELECT email FROM users WHERE email=?", (fp_email,))
            if not cur.fetchone():
                st.error("No such user")
            else:
                otp_fp = gen_otp()
                save_otp(fp_email, otp_fp)
                if SMTP_EMAIL and SMTP_APP_PASSWORD:
                    send_email_html(fp_email, "Password Reset OTP", f"<b>{otp_fp}</b>")
                st.info("OTP sent (or shown in dev mode).")
        st.markdown("---")
        rp_email = st.text_input("Email for reset", key="rp_email")
        rp_otp = st.text_input("OTP", key="rp_otp")
        rp_new = st.text_input("New password", type="password", key="rp_new")
        if st.button("Reset Password"):
            if not verify_otp(rp_email, rp_otp):
                st.error("Invalid/expired OTP")
            else:
                new_h = bcrypt.hashpw(rp_new.encode(), bcrypt.gensalt())
                cur.execute("UPDATE users SET password_hash=? WHERE email=?", (new_h, rp_email))
                con.commit()
                st.success("Password updated. You can login now.")
        st.markdown("</div>", unsafe_allow_html=True)

# LOGGED-IN ROUTES
else:
    if nav == "Monitor" or st.session_state.get("show_monitor"):
        st.header("Monitor")
        col_left, col_right = st.columns([3,1])
        with col_right:
            st.markdown("### Controls")
            st.write(f"Crowd threshold: **{settings.get('crowd_threshold')}**")
            st.write(f"Weapon duration(s): **{settings.get('weapon_duration')}**")
            st.write(f"Violence duration(s): **{settings.get('violence_duration')}**")
            st.write(f"Alert cooldown(s): **{settings.get('alert_cooldown')}**")
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.show_monitor = False
                try:
                    if os.path.exists(SESSION_TOKEN_FILE):
                        os.remove(SESSION_TOKEN_FILE)
                except:
                    pass
                st.rerun()
        with col_left:
            mode = st.radio("Stream Source", ["Webcam (Live)", "Upload Video"], horizontal=True)
            def get_rtc_configuration():
                    # Multiple STUN servers improves connection success
                    ice_servers = [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:global.stun.twilio.com:3478?transport=udp"]},
                    ]
                    return {"iceServers": ice_servers}
                
                
            if mode == "Webcam (Live)":
                    st.info("Start your webcam (webrtc). Alerts will be sent when rules fire.")
                
                    rtc_conf = get_rtc_configuration()
                
                    try:
                        webrtc_streamer(
                            key="live_stream",
                            rtc_configuration=rtc_conf,
                            video_transformer_factory=DetectorTransformer,
                            media_stream_constraints={"video": True, "audio": False},
                            async_transform=True,
                        )
                    except Exception as e:
                        st.error(
                            "Live stream failed to start. This can happen when WebRTC cannot establish a connection "
                            "(STUN/TURN problem or network restrictions)."
                        )
                        st.write("Error (hidden):", str(e))
                        st.info(
                            "Try:\n"
                            "• Adding a TURN server (recommended) and put its username/credential in Streamlit Secrets.\n"
                            "• Or try from a different network (some networks block UDP)."
                        )
            else:
                st.info("Upload a video file (mp4) to process.")
                uploaded = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"])
                if uploaded is not None:
                    path = Path("uploaded_video.mp4")
                    path.write_bytes(uploaded.getbuffer())
                    cap = cv2.VideoCapture(str(path))
                    transformer = DetectorTransformer()
                    placeholder = st.empty()
                    fps = cap.get(cv2.CAP_PROP_FPS) or 24
                    delay = 1.0 / fps
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        class F:
                            def to_ndarray(self, format="bgr24"):
                                return frame
                        out = transformer.transform(F())
                        placeholder.image(out, channels="RGB", use_container_width=True)
                        time.sleep(delay)
                    cap.release()
                    placeholder.empty()
                    st.success("Finished processing uploaded video.")

    elif nav == "Logout":
        st.session_state.user = None
        st.session_state.show_monitor = False
        try:
            if os.path.exists(SESSION_TOKEN_FILE):
                os.remove(SESSION_TOKEN_FILE)
        except:
            pass
        st.rerun()

# Footer
st.markdown("<div style='padding:14px; text-align:center; color:#9aa7b8; margin-top:18px;'>©Smart Detection System </div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center;">
    <a href="https://t.me/corwd_msg_bot" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/8/82/Telegram_logo.svg" width="80">
    </a>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='padding:14px; text-align:center; color:#9aa7b8; margin-top:18px;'>Click on Icon to get notified by telegram </div>", unsafe_allow_html=True)







