
import base64
import csv
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request, jsonify
from torchvision import transforms

# ==========================
# CONFIG
# ==========================

app = Flask(__name__)

PROJECT_ROOT = Path(".").resolve()

# Emotion + liveness model weights
EMO_MODEL_PATH = PROJECT_ROOT / "WEIGHTS_GROUP" / "dual_head_raf_emo.pth"
LIVE_MODEL_PATH = PROJECT_ROOT / "WEIGHTS_GROUP" / "dual_head_samm_liveness.pth"

# CSV log
AUDIT_CSV = PROJECT_ROOT / "liveness_audit.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Email settings
SMTP_HOST = "smtp.gmail.com"             
SMTP_PORT = 587
SMTP_USER = "xyz@gmail.com"      
SMTP_PASSWORD = "disclosed"  


# RAF-style emotions
EMOTION_CLASSES = [
    "Surprise",   # 1
    "Fear",       # 2
    "Disgust",    # 3
    "Happiness",  # 4
    "Sadness",    # 5
    "Anger",      # 6
    "Neutral",    # 7
]


LIVENESS_CLASSES = ["live", "spoof"]

# Common transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Haar face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================
# MODEL
# ==========================

class DualHeadViT(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        num_emotions: int = 7,
        num_liveness: int = 2,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,
            global_pool="avg",
        )
        embed_dim = self.backbone.num_features
        self.emotion_head = nn.Linear(embed_dim, num_emotions)
        self.liveness_head = nn.Linear(embed_dim, num_liveness)

    def forward(self, x):
        feats = self.backbone(x)
        emo_logits = self.emotion_head(feats)
        live_logits = self.liveness_head(feats)
        return emo_logits, live_logits


# ==========================
# PREPROCESSING
# ==========================

def preprocess_full_frame(bgr_img):
    """
    For liveness model:
    center-square crop, resize, normalize → tensor [1, 3, H, W]
    """
    h, w, _ = bgr_img.shape
    side = min(h, w)

    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    crop = bgr_img[y1:y1 + side, x1:x1 + side]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor


def preprocess_face_crop(bgr_img, margin: float = 0.3):
    """
    For emotion model:
    detect largest face, crop with margin, resize, normalize.
    Fallback: center-square crop if no face detected.
    Returns tensor [1, 3, H, W].
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    h0, w0, _ = bgr_img.shape

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx = x + w // 2
        cy = y + h // 2

        new_w = int(w * (1 + margin))
        new_h = int(h * (1 + margin))

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w0, cx + new_w // 2)
        y2 = min(h0, cy + new_h // 2)

        crop = bgr_img[y1:y2, x1:x2]
    else:
 
        side = min(h0, w0)
        cx, cy = w0 // 2, h0 // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        crop = bgr_img[y1:y1 + side, x1:x1 + side]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor


# ==========================
# EMAIL + CSV HELPERS
# ==========================

def send_spoof_alert(to_email: str, emotion: str, raw_live: float, raw_spoof: float):
    """
    Send an email alert when a spoof login attempt is confirmed.
    Email is sent to the user who attempted to log in.
    """
    if not to_email:
        print("[EMAIL] No recipient email provided, skipping alert.")
        return

    subject = "AuthentiFace Alert: Possible spoof login attempt"
    body = (
        "A possible spoof login attempt was detected on your account.\n\n"
        f"Details:\n"
        f"- Emotion detected: {emotion}\n"
        f"- Raw P(live):  {raw_live:.2f}\n"
        f"- Raw P(spoof): {raw_spoof:.2f}\n\n"
        "If this was not you, please review your account security immediately.\n\n"
        "— AuthentiFace Security"
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_email

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"[EMAIL] Spoof alert sent to '{to_email}'")
    except Exception as e:
        # Don't crash the app if email fails
        print(f"[EMAIL] Failed to send spoof alert: {e}")


def append_audit_row(
    email: str,
    decision: str,
    reason: str,
    avg_live: float,
    avg_spoof: float,
    raw_live: float,
    raw_spoof: float,
    emotion: str,
):
    """
    One CSV row per completed attempt (granted or denied).
    Separate date & time columns. Logs email (used as identity).
    """
    date_str = time.strftime("%Y-%m-%d", time.localtime())
    time_str = time.strftime("%H:%M:%S", time.localtime())

    row = {
        "date": date_str,
        "time": time_str,
        "email": email or "",
        "decision": decision,
        "reason": reason,
        "avg_live": f"{avg_live:.4f}",
        "avg_spoof": f"{avg_spoof:.4f}",
        "raw_live": f"{raw_live:.4f}",
        "raw_spoof": f"{raw_spoof:.4f}",
        "emotion": emotion,
    }

    file_exists = AUDIT_CSV.exists()

    with AUDIT_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "time",
                "email",
                "decision",
                "reason",
                "avg_live",
                "avg_spoof",
                "raw_live",
                "raw_spoof",
                "emotion",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[CSV] Logged {decision} for email '{email}'")


# ==========================
# INIT MODELS
# ==========================

print("Loading emotion and liveness models for API...")

if not EMO_MODEL_PATH.exists():
    raise FileNotFoundError(f"Emotion model weights not found at {EMO_MODEL_PATH}")
if not LIVE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Liveness model weights not found at {LIVE_MODEL_PATH}")

# Emotion model (RAF)
emotion_model = DualHeadViT(
    backbone_name="vit_tiny_patch16_224",
    num_emotions=len(EMOTION_CLASSES),
    num_liveness=len(LIVENESS_CLASSES),
    pretrained_backbone=False,
).to(DEVICE)
emotion_state = torch.load(EMO_MODEL_PATH, map_location=DEVICE)
emotion_model.load_state_dict(emotion_state, strict=False)
emotion_model.eval()

# Liveness model (CelebA-Spoof dual-head)
liveness_model = DualHeadViT(
    backbone_name="vit_tiny_patch16_224",
    num_emotions=len(EMOTION_CLASSES),
    num_liveness=len(LIVENESS_CLASSES),
    pretrained_backbone=False,
).to(DEVICE)
liveness_state = torch.load(LIVE_MODEL_PATH, map_location=DEVICE)
liveness_model.load_state_dict(liveness_state, strict=False)
liveness_model.eval()

print("Emotion model loaded from:", EMO_MODEL_PATH)
print("Liveness model loaded from:", LIVE_MODEL_PATH)
print("Device:", DEVICE)


# ==========================
# ROUTES
# ==========================

@app.route("/")
def index():

    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    user_email = request.form.get("email", "")
    _password = request.form.get("password", "")




    return render_template("liveness.html", email=user_email)


@app.route("/success")
def success():
    email = request.args.get("email", "")
    emotion = request.args.get("emotion", "Unknown")
    return render_template("success.html", email=email, emotion=emotion)


@app.route("/denied")
def denied():
    email = request.args.get("email", "")
    emotion = request.args.get("emotion", "Unknown")
    return render_template("denied.html", email=email, emotion=emotion)


# ---------- MAIN INFERENCE API ----------
@app.route("/api/liveness", methods=["POST"])
def api_liveness():
    """
    Receives JSON: { "image": "data:image/jpeg;base64,...." }
    - Liveness: full-frame → liveness_model
    - Emotion: face crop → emotion_model

    Returns:
    {
      prob_live,
      prob_spoof,
      emotion,
      label
    }
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    img_str = data["image"]


    if img_str.startswith("data:image"):
        img_str = img_str.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(img_str)
    except Exception as e:
        return jsonify({"error": f"Invalid base64: {e}"}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Failed to decode image"}), 400


    full_tensor = preprocess_full_frame(frame).to(DEVICE)
    face_tensor = preprocess_face_crop(frame).to(DEVICE)

    with torch.no_grad():
        # Emotion prediction
        emo_logits, _ = emotion_model(face_tensor)
        emo_probs = emo_logits.softmax(dim=1)[0].cpu().numpy()
        emo_idx = int(np.argmax(emo_probs))
        emotion = EMOTION_CLASSES[emo_idx]

        # Liveness prediction
        _, live_logits = liveness_model(full_tensor)
        live_probs = live_logits.softmax(dim=1)[0].cpu().numpy()

    prob_live = float(live_probs[0])   # index 0 = 'live'
    prob_spoof = float(live_probs[1])  # index 1 = 'spoof'
    label = "live" if prob_live >= prob_spoof else "spoof"

    return jsonify({
        "prob_live": prob_live,
        "prob_spoof": prob_spoof,
        "emotion": emotion,
        "label": label,
    }), 200


# ---------- LOGGING APIs ----------
@app.route("/api/log_live", methods=["POST"])
def api_log_live():
    """
    Frontend calls this when live is confirmed.
    Logs to CSV (no email).
    """
    data = request.get_json(force=True) or {}
    user_email = data.get("email", "")
    avg_live = float(data.get("avg_live", 0.0))
    avg_spoof = float(data.get("avg_spoof", 0.0))
    raw_live = float(data.get("raw_live", 0.0))
    raw_spoof = float(data.get("raw_spoof", 0.0))
    emotion = data.get("emotion", "Unknown")

    append_audit_row(
        email=user_email,
        decision="granted",
        reason="client_rule: live",
        avg_live=avg_live,
        avg_spoof=avg_spoof,
        raw_live=raw_live,
        raw_spoof=raw_spoof,
        emotion=emotion,
    )
    return jsonify({"status": "ok"})


@app.route("/api/log_spoof", methods=["POST"])
def api_log_spoof():
    """
    Frontend calls this when spoof is confirmed (3s window).
    Logs to CSV + sends email alert to the user who attempted login.
    """
    data = request.get_json(force=True) or {}
    user_email = data.get("email", "")
    avg_live = float(data.get("avg_live", 0.0))
    avg_spoof = float(data.get("avg_spoof", 0.0))
    raw_live = float(data.get("raw_live", 0.0))
    raw_spoof = float(data.get("raw_spoof", 0.0))
    emotion = data.get("emotion", "Unknown")

    append_audit_row(
        email=user_email,
        decision="denied",
        reason="client_rule: spoof",
        avg_live=avg_live,
        avg_spoof=avg_spoof,
        raw_live=raw_live,
        raw_spoof=raw_spoof,
        emotion=emotion,
    )

  
    send_spoof_alert(
        to_email=user_email,
        emotion=emotion,
        raw_live=raw_live,
        raw_spoof=raw_spoof,
    )

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
