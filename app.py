# streamlit_app.py
"""
Clean Streamlit CCTV app prepared for Render deployment.
All Windows paths removed and replaced with relative paths.
Environment variables read using os.getenv (Render-friendly).
All functionalities preserved: OTP, Email, Telegram, Alerts, YOLO, WebRTC.
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
# CONFIG / ENV (Render Compatible)
# -------------------------

# These must be added in Render → Environment Variables
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Model paths (relative paths inside repo)
CROWD_MODEL_PATH = os.getenv("CROWD_MODEL_PATH", "People_count_model/best.pt")
WEAPON_MODEL_PATH = os.getenv("WEAPON_MODEL_PATH", "weapon_detection_model/best1.pt")

DB_PATH = "users.db"
SETTINGS_PATH = "settings.json"
SESSION_TOKEN_FILE = ".session_token"

# LOCAL ASSET (works on Render)
HERO_IMG = "assets/ai_mon.jpg"   # MUST exist inside repo

# Default settings
DEFAULT_SETTINGS = {
    "crowd_threshold": 1000,
    "weapon_duration": 3.0,
    "violence_duration": 3.0,
    "alert_cooldown": 20
}

def load_settings():
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r") as f:
                data = json.load(f)
            for k,v in DEFAULT_SETTINGS.items():
                if k not in data:
                    data[k] = v
            return data
        except:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)

settings = load_settings()

# -------------------------
# DATABASE (sqlite)
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

cur.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT
)
""")

con.commit()

def safe_add_column(table, col):
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
        con.commit()
    except:
        pass

safe_add_column("alerts","type TEXT")
safe_add_column("alerts","details TEXT")
safe_add_column("alerts","created_at INTEGER")
safe_add_column("alerts","snapshot_path TEXT")

# -------------------------
# EMAIL + TELEGRAM
# -------------------------

def send_email_html(to_email, subject, html, text=None):
    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        print("Email not configured")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email
        if text:
            msg.set_content(text)
        msg.add_alternative(html, subtype="html")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            s.send_message(msg)
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
        requests.post(url, data={"chat_id":TELEGRAM_CHAT_ID,"text":text})
        return True
    except:
        return False

# -------------------------
# OTP + AUTH
# -------------------------

def gen_otp():
    return "".join(random.choices(string.digits, k=6))

def save_otp(email, otp):
    cur.execute("INSERT INTO otps (email, otp, created_at) VALUES (?,?,?)",
                (email, otp, int(time.time())))
    con.commit()

def verify_otp(email, otp):
    cur.execute("SELECT otp, created_at FROM otps WHERE email=? ORDER BY created_at DESC LIMIT 1",(email,))
    row = cur.fetchone()
    if not row: return False
    saved, ts = row
    if time.time() - ts > 300: return False
    return saved == otp

def register_user(email,pwd):
    try:
        v = validate_email(email)
        email = v.email
    except Exception as e:
        return False, str(e)

    h = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt())
    try:
        cur.execute("INSERT OR REPLACE INTO users (email,password_hash,verified) VALUES (?,?,0)",
                    (email, h))
        con.commit()
        return True, "Registered"
    except Exception as e:
        return False, str(e)

def check_login(email,pwd):
    cur.execute("SELECT password_hash,verified FROM users WHERE email=?",(email,))
    row = cur.fetchone()
    if not row: return False, "Account not found"
    h, verified = row
    if not bcrypt.checkpw(pwd.encode(), h):
        return False, "Wrong password"
    if verified == 0:
        return False, "Account not verified"
    return True, ""


# -------------------------
# YOLO MODELS
# -------------------------
try:
    from ultralytics import YOLO
except:
    YOLO = None

@st.cache_resource
def load_models():
    crowd = YOLO(CROWD_MODEL_PATH) if YOLO else None
    weapon = YOLO(WEAPON_MODEL_PATH) if YOLO else None
    return crowd, weapon

crowd_model, weapon_model = load_models()

# -------------------------
# Video transformer (WebRTC)
# -------------------------

class DetectorTransformer(VideoTransformerBase):

    def __init__(self):
        self.last_alert = 0
        self.first_seen = {"weapon":None,"violence":None}
        self.alerted = {"weapon":False,"violence":False}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        vis = img.copy()
        now = time.time()

        crowd_count = 0
        weapon_detected = False
        violence_detected = False
        labels_found = []

        # CROWD
        if crowd_model:
            try:
                r = crowd_model(img, verbose=False)
                if len(r) > 0:
                    boxes = r[0].boxes.xyxy.cpu().numpy()
                    crowd_count = len(boxes)
                    for x1,y1,x2,y2 in boxes:
                        cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            except:
                pass

        # WEAPON & VIOLENCE
        if weapon_model:
            try:
                rs = weapon_model(img, conf=0.45)
                if len(rs) > 0:
                    r = rs[0]
                    if hasattr(r,"boxes"):
                        for box in r.boxes:
                            cls = int(box.cls)
                            conf = float(box.conf)
                            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                            name = weapon_model.names[cls].lower()

                            if "violence" in name:
                                violence_detected = True
                                cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3)
                                cv2.putText(vis,"Violence",(int(x1),int(y1)-5),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

                            elif any(w in name for w in ["gun","knife","weapon"]):
                                weapon_detected = True
                                labels_found.append(name)
                                cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3)
                                cv2.putText(vis,"Weapon",(int(x1),int(y1)-5),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            except:
                pass

        # HEADER
        cv2.rectangle(vis,(0,0),(vis.shape[1],50),(0,0,0),-1)
        cv2.putText(vis, f"Crowd:{crowd_count} | Weapon:{weapon_detected} | Violence:{violence_detected}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255),2)

        # ALERT RULES
        crowd_threshold = settings["crowd_threshold"]
        wd = settings["weapon_duration"]
        vd = settings["violence_duration"]
        cooldown = settings["alert_cooldown"]

        # CROWD
        if crowd_count > crowd_threshold and (now - self.last_alert > cooldown):
            msg = f"🚨 CROWD ALERT 🚨\nCrowd={crowd_count}\nTime={time.ctime()}"
            send_telegram_alert(msg)
            cur.execute("SELECT email FROM users WHERE verified=1")
            rows = cur.fetchall()
            if rows:
                send_email_html(",".join([r[0] for r in rows]),"Crowd Alert",msg,msg)
            self.last_alert = now

        # WEAPON
        if weapon_detected:
            if self.first_seen["weapon"] is None:
                self.first_seen["weapon"] = now
            if (now - self.first_seen["weapon"] >= wd) and not self.alerted["weapon"] and (now - self.last_alert > cooldown):
                msg = f"🚨 WEAPON ALERT 🚨\nDetected={labels_found}\nTime={time.ctime()}"
                send_telegram_alert(msg)
                cur.execute("SELECT email FROM users WHERE verified=1")
                rows = cur.fetchall()
                if rows:
                    send_email_html(",".join([r[0] for r in rows]),"Weapon Alert",msg,msg)
                self.alerted["weapon"] = True
                self.last_alert = now
        else:
            self.first_seen["weapon"] = None
            self.alerted["weapon"] = False

        # VIOLENCE
        if violence_detected:
            if self.first_seen["violence"] is None:
                self.first_seen["violence"] = now
            if (now - self.first_seen["violence"] >= vd) and not self.alerted["violence"] and (now - self.last_alert > cooldown):
                msg = f"🚨 VIOLENCE ALERT 🚨\nTime={time.ctime()}"
                send_telegram_alert(msg)
                cur.execute("SELECT email FROM users WHERE verified=1")
                rows = cur.fetchall()
                if rows:
                    send_email_html(",".join([r[0] for r in rows]),"Violence Alert",msg,msg)
                self.alerted["violence"] = True
                self.last_alert = now
        else:
            self.first_seen["violence"] = None
            self.alerted["violence"] = False

        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Smart CCTV", layout="wide")

# CSS
st.markdown("""
<style>
body {
    background: #030615;
}
.card { background:#0f1724; padding:20px; border-radius:10px; margin:20px;}
</style>
""", unsafe_allow_html=True)

# session vars
if "user" not in st.session_state: st.session_state.user = None
if "just_registered" not in st.session_state: st.session_state.just_registered = None
if "show_forgot" not in st.session_state: st.session_state.show_forgot = False

# SIDEBAR
with st.sidebar:
    st.title("Navigation")
    if st.session_state.user:
        nav = st.radio("",["Monitor","Logout"])
    else:
        nav = st.radio("",["Home","Register","Login","Settings"])

if nav=="Logout":
    st.session_state.user = None
    st.session_state.show_monitor = False
    st.rerun()

# -------------------------
# ROUTES (LOGGED OUT)
# -------------------------

if not st.session_state.user:

    if nav=="Home":
        st.header("Real-Time Crowd, Weapon & Violence Detection")
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("System Features")
            st.write("""
            - Crowd Counting  
            - Weapon Detection (Gun/Knife)  
            - Violence Detection  
            - Automatic Alerts (Email + Telegram)  
            - Live WebRTC Streaming  
            """)
        with col2:
            st.image(HERO_IMG, use_container_width=True)

    if nav=="Register":
        st.subheader("Register")
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        if st.button("Register"):
            ok,msg = register_user(email,pwd)
            if not ok:
                st.error(msg)
            else:
                otp = gen_otp()
                save_otp(email,otp)
                if SMTP_EMAIL:
                    send_email_html(email,"OTP Verification",f"<b>{otp}</b>")
                st.session_state.just_registered = email
                st.success("OTP sent to email")

        if st.session_state.just_registered:
            otp = st.text_input("Enter OTP")
            if st.button("Verify"):
                if verify_otp(st.session_state.just_registered, otp):
                    cur.execute("UPDATE users SET verified=1 WHERE email=?",(st.session_state.just_registered,))
                    con.commit()
                    st.session_state.user = st.session_state.just_registered
                    st.session_state.just_registered = None
                    st.success("Verified! Redirecting...")
                    st.rerun()
                else:
                    st.error("Wrong/Expired OTP")

    if nav=="Login":
        st.subheader("Login")
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            ok,msg = check_login(email,pwd)
            if ok:
                st.session_state.user = email
                st.success("Logged in")
                st.rerun()
            else:
                st.error(msg)

    if nav=="Settings":
        st.subheader("Public Settings")
        ct = st.number_input("Crowd threshold",min_value=1,value=settings["crowd_threshold"])
        wd = st.number_input("Weapon duration (sec)",min_value=0.5,value=settings["weapon_duration"])
        vd = st.number_input("Violence duration (sec)",min_value=0.5,value=settings["violence_duration"])
        ac = st.number_input("Alert cooldown (sec)",min_value=1,value=settings["alert_cooldown"])
        if st.button("Save"):
            settings["crowd_threshold"] = int(ct)
            settings["weapon_duration"] = float(wd)
            settings["violence_duration"] = float(vd)
            settings["alert_cooldown"] = int(ac)
            save_settings(settings)
            st.success("Saved.")

# -------------------------
# ROUTES (LOGGED IN)
# -------------------------

else:

    if nav=="Monitor":
        st.header("Live Monitor")

        mode = st.radio("Source",["Webcam (Live)","Upload Video"],horizontal=True)

        if mode=="Webcam (Live)":
            rtc_conf = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
            webrtc_streamer(
                key="live",
                video_transformer_factory=DetectorTransformer,
                rtc_configuration=rtc_conf,
                media_stream_constraints={"video":True,"audio":False},
                async_transform=True
            )

        else:
            file = st.file_uploader("Upload video",type=["mp4","avi","mov","mkv"])
            if file:
                path = "uploaded_video.mp4"
                open(path,"wb").write(file.getbuffer())
                cap = cv2.VideoCapture(path)
                tf = DetectorTransformer()
                ph = st.empty()
                fps = cap.get(cv2.CAP_PROP_FPS) or 24
                wait = 1/fps
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    class Tmp:
                        def to_ndarray(self,format="bgr24"):
                            return frame
                    out = tf.transform(Tmp())
                    ph.image(out,channels="RGB")
                    time.sleep(wait)
                cap.release()
                st.success("Done.")

