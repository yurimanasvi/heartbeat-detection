import streamlit as st
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
import random
import time

st.set_page_config(page_title="Disaster Heartbeat Detection", page_icon="🫀", layout="wide")

st.markdown('''
<style>
    .room-card { border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 8px; border: 2px solid transparent; }
    .room-victim    { background: #1a0a0a; border-color: #ff4444; }
    .room-clear     { background: #0a1a0a; border-color: #44cc44; }
    .room-scanning  { background: #1a1a0a; border-color: #ffaa00; }
    .room-unknown   { background: #1a1a1a; border-color: #555555; }
    .stat-box { background: #1a1a2e; border-radius: 8px; padding: 0.75rem; text-align: center; border: 1px solid #333; }
    .stat-val { font-size: 28px; font-weight: 700; }
    .stat-lbl { font-size: 12px; color: #aaa; margin-top: 2px; }
    .alert-box { background: #2a0a0a; border: 1px solid #ff4444; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 8px; font-size: 13px; }
    .warn-box  { background: #2a1a0a; border: 1px solid #ffaa00; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 8px; font-size: 13px; }
</style>
''', unsafe_allow_html=True)

def generate_heartbeat(duration=5, sr=100, bpm=None, strength=1.0):
    if bpm is None:
        bpm = random.uniform(55, 130)
    hz = bpm / 60
    t  = np.linspace(0, duration, duration * sr)
    hb = (np.sin(2*np.pi*hz*t) + 0.3*np.sin(2*np.pi*2*hz*t) + 0.1*np.sin(2*np.pi*3*hz*t)) * strength
    hb += np.random.normal(0, 0.05*strength, len(t))
    return t, hb

def add_noise(signal, level=1.0, sr=100):
    n   = len(signal)
    t   = np.linspace(0, n/sr, n)
    out = signal + np.random.normal(0, level, n)
    out += 0.5 * np.sin(2*np.pi*0.3*t)
    out += 0.3 * np.sin(2*np.pi*15*t) * (np.random.rand(n) > 0.7)
    return out

def bandpass(signal, lo=0.4, hi=3.5, sr=100):
    nyq  = sr / 2
    b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, signal)

def get_features(signal, sr=100):
    fft_v = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    hb_m  = (freqs >= 0.5) & (freqs <= 3.0)
    hb_p  = np.sum(fft_v[hb_m]**2)
    ns_p  = np.sum(fft_v[~hb_m]**2) + 1e-10
    fft_n = fft_v / (np.sum(fft_v) + 1e-10)
    return [np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
            np.ptp(signal), np.sum(np.abs(np.diff(signal))),
            hb_p, ns_p, hb_p/ns_p, freqs[np.argmax(fft_v)],
            -np.sum(fft_n * np.log(fft_n + 1e-10)), np.sum(fft_v[hb_m]**2)]

@st.cache_resource
def train_model():
    X, y = [], []
    # Normal heartbeats
    for _ in range(300):
        _, hb = generate_heartbeat(bpm=random.uniform(35,140), strength=random.uniform(0.3,1.0))
        sig   = add_noise(hb, random.uniform(0.6,1.5))
        X.append(get_features(bandpass(sig))); y.append(1)
    # WEAK heartbeats — extra training so model learns to detect them
    for _ in range(200):
        _, hb = generate_heartbeat(bpm=random.uniform(35,60), strength=random.uniform(0.05,0.25))
        sig   = add_noise(hb, random.uniform(0.8,1.5))
        X.append(get_features(bandpass(sig))); y.append(1)
    # Noise / empty rooms
    for _ in range(500):
        sig = add_noise(np.zeros(500), random.uniform(0.6,1.5))
        X.append(get_features(bandpass(sig))); y.append(0)
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    clf.fit(np.array(X), np.array(y))
    return clf, 1000, 0

def scan_room(model, has_victim=True, bpm=72, strength=1.0, noise=1.0):
    # Scan 3 times and take majority vote for better accuracy
    preds, confs = [], []
    for _ in range(5):
        if has_victim:
            _, hb = generate_heartbeat(bpm=bpm, strength=strength)
        else:
            hb = np.zeros(500)
        sig  = add_noise(hb, noise)
        filt = bandpass(sig)
        feat = np.array([get_features(filt)])
        preds.append(model.predict(feat)[0])
        confs.append(max(model.predict_proba(feat)[0]))
    # Majority vote
    final_pred = 1 if sum(preds) >= 3 else 0
    final_conf = float(np.mean(confs))
    return final_pred, final_conf

ROOMS = [
    ("R101","Room 101",    1,True, 78,  0.90,0.8,"Normal adult"),
    ("R102","Room 102",    1,False,None,0.00,0.9,"Empty"),
    ("R103","Room 103",    1,True, 42,  0.18,1.2,"Injured (weak)"),
    ("R104","Room 104",    1,True, 118, 0.75,1.0,"Panicking"),
    ("R105","Stairwell A", 1,False,None,0.00,1.3,"Empty"),
    ("R201","Room 201",    2,True, 65,  0.85,0.7,"Normal adult"),
    ("R202","Room 202",    2,False,None,0.00,0.8,"Empty"),
    ("R203","Room 203",    2,True, 95,  0.60,1.1,"Child"),
    ("R204","Room 204",    2,False,None,0.00,1.0,"Empty"),
    ("R205","Stairwell B", 2,True, 38,  0.15,1.4,"Injured (very weak)"),
]

if "results"    not in st.session_state: st.session_state.results    = {}
if "scanning"   not in st.session_state: st.session_state.scanning   = False
if "scan_log"   not in st.session_state: st.session_state.scan_log   = []
if "model"      not in st.session_state: st.session_state.model      = None
if "synth_count" not in st.session_state: st.session_state.synth_count = 0

st.markdown("# 🫀 Disaster Heartbeat Detection System")
st.markdown("AI-powered victim detection · Week 4: Real ECG Data")
st.markdown("---")

with st.sidebar:
    st.markdown("### Control Panel")
    st.markdown("---")
    if st.button("🔄 Load & train AI model", use_container_width=True):
        with st.spinner("Training…"):
            model, synth_count, real_count = train_model()
            st.session_state.model       = model
            st.session_state.synth_count = synth_count
        st.success("✅ Model ready!")
    if st.session_state.model is not None:
        st.markdown(f"🧪 Training samples: `{st.session_state.synth_count}`")
        st.markdown("✅ Weak heartbeat detection ON")
    st.markdown("---")
    scan_floor = st.selectbox("Floor to scan", ["All floors","Floor 1","Floor 2"])
    scan_speed = st.slider("Scan speed (sec per room)", 0.3, 2.0, 0.6, step=0.1)
    if st.button("▶ Start scan", use_container_width=True,
                 disabled=(st.session_state.model is None)):
        st.session_state.scanning = True
        st.session_state.results  = {}
        st.session_state.scan_log = []
    if st.button("⏹ Reset", use_container_width=True):
        st.session_state.results  = {}
        st.session_state.scan_log = []
        st.session_state.scanning = False
    st.markdown("---")
    st.markdown("🔴 Victim detected")
    st.markdown("🟢 Clear")
    st.markdown("🟡 Uncertain — rescan")
    st.markdown("⚪ Not scanned yet")

if st.session_state.get("scanning") and st.session_state.get("model"):
    target = [r for r in ROOMS if scan_floor=="All floors" or
              (scan_floor=="Floor 1" and r[2]==1) or
              (scan_floor=="Floor 2" and r[2]==2)]
    for room in target:
        rid,name,floor,has_v,bpm,strength,noise,desc = room
        time.sleep(scan_speed)
        pred,conf = scan_room(st.session_state.model,
                              has_victim=has_v, bpm=bpm or 72,
                              strength=strength, noise=noise)
        if pred==1 and conf>=0.50:    status="victim"
        elif pred==1 and conf<0.50:   status="uncertain"
        elif pred==0 and conf<0.65:   status="uncertain"
        else:                         status="clear"
        st.session_state.results[rid] = {"status":status,"conf":conf,"name":name,"desc":desc}
        if status=="victim":      log=f"🔴 {name} — VICTIM DETECTED ({conf*100:.0f}%)"
        elif status=="uncertain": log=f"🟡 {name} — UNCERTAIN, manual check ({conf*100:.0f}%)"
        else:                     log=f"🟢 {name} — Clear ({conf*100:.0f}%)"
        st.session_state.scan_log.append(log)
    st.session_state.scanning = False

results = st.session_state.results
if results:
    v = sum(1 for r in results.values() if r["status"]=="victim")
    u = sum(1 for r in results.values() if r["status"]=="uncertain")
    c = sum(1 for r in results.values() if r["status"]=="clear")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#ff4444;">{v}</div><div class="stat-lbl">Victims found</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#ffaa00;">{u}</div><div class="stat-lbl">Uncertain</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#44cc44;">{c}</div><div class="stat-lbl">Clear</div></div>',unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#aaa;">{len(results)}/10</div><div class="stat-lbl">Scanned</div></div>',unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def card(rid, data):
    s = data.get("status","unknown")
    if s=="victim":     css,icon,label="room-victim","🔴","VICTIM DETECTED"
    elif s=="clear":    css,icon,label="room-clear","🟢","CLEAR"
    elif s=="uncertain":css,icon,label="room-scanning","🟡","UNCERTAIN"
    else:               css,icon,label="room-unknown","⚪","NOT SCANNED"
    conf_txt = f"{data.get('conf',0)*100:.0f}% confidence" if s not in ("unknown",) else ""
    return f'<div class="room-card {css}"><div style="font-weight:600;">{data.get("name",rid)}</div><div style="font-size:11px;color:#888;">{data.get("desc","")}</div><div style="font-size:22px;margin:6px 0;">{icon}</div><div style="font-size:13px;font-weight:600;">{label}</div><div style="font-size:12px;color:#aaa;">{conf_txt}</div></div>'

for floor_num in [1,2]:
    st.markdown(f"#### Floor {floor_num}")
    floor_rooms = [r for r in ROOMS if r[2]==floor_num]
    cols = st.columns(len(floor_rooms))
    for col,room in zip(cols,floor_rooms):
        rid  = room[0]
        data = results.get(rid,{"status":"unknown","conf":0,"name":room[1],"desc":room[7]})
        with col: st.markdown(card(rid,data), unsafe_allow_html=True)
    st.markdown("---")

if st.session_state.scan_log:
    st.markdown("#### Rescue Alert Log")
    for entry in reversed(st.session_state.scan_log):
        if "VICTIM"      in entry: st.markdown(f'<div class="alert-box">{entry}</div>',unsafe_allow_html=True)
        elif "UNCERTAIN" in entry: st.markdown(f'<div class="warn-box">{entry}</div>',unsafe_allow_html=True)
        else: st.markdown(f'<div style="font-size:13px;color:#aaa;padding:4px 0;">{entry}</div>',unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center;font-size:12px;color:#aaa;">Heartbeat Disaster Detection · College Project · Week 4</p>', unsafe_allow_html=True)
