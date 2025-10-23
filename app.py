import os
import cv2
import numpy as np
import torch
import timm
from mtcnn import MTCNN
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, Response, redirect, url_for, jsonify, request, session
import threading
import yagmail
import torch.nn.functional as F
from datetime import datetime
import time

REAL_CLASS_IDX = 0
app = Flask(__name__)
app.secret_key = 'supersecretkey'  

detector = MTCNN()

model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 2)  
pretrained_model_path = 'weights/vit_teacher_inc_reduced_lr-7.pth'
if not os.path.exists(pretrained_model_path):
    raise FileNotFoundError(f"Pretrained model file not found: {pretrained_model_path}")

model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
model.eval()

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")

face_cascade = cv2.CascadeClassifier(cascade_path)

SENDER_EMAIL = 'akshay.dhoni02@gmail.com'
APP_PASSWORD = 'oqwm mjds rcki lpho'  
RECEIVER_EMAIL = 'akshay.suresh155@gmail.com'

def send_email(subject, body):
    yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
    print(f"[EMAIL] Sent: {subject} → {RECEIVER_EMAIL}")
    try:
        yag.send(to=RECEIVER_EMAIL, subject=subject, contents=body)
    except Exception as e:
        print(f"Error sending email: {e}")

camera = None
processing = False
result_data = {'result': None, 'color': None}
recorded_frames = []
email_sent = False
frame_count = 0

def start_video_capture():
    global camera
    camera_index = 0 
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not start camera at index {camera_index}. Please check if the camera is connected and accessible.")

def stop_video_capture():
    global camera
    if camera:
        camera.release()

def capture_frames(account_number):
    global camera, processing, result_data, recorded_frames, email_sent, frame_count

    # counters
    frame_count = 0
    real_count = 0
    fake_count = 0
    recorded_frames = []

    # timing / stopping rules
    MIN_FRAMES = 60          # collect at least ~2–3s worth of frames
    MAX_SECONDS = 5          # or stop after 5 seconds, whichever comes first
    start_time = None

    # read frames until we have enough evidence or timeout
    while not processing:
        if camera and camera.isOpened():
            success, frame = camera.read()
            if not success:
                continue

            # first timestamp
            if start_time is None:
                start_time = time.time()

            # detect faces
            boxes = detector.detect_faces(frame)

            # require exactly one face (skip frames with 0 or >1)
            if len(boxes) != 1:
                current_time = time.time()
                if start_time and (current_time - start_time) >= MAX_SECONDS:
                    break
                continue

            # take the highest-confidence face
            boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
            x, y, w, h = boxes[0]['box']

            # preprocess face and predict probability of REAL
            face_tensor = process_face(frame, x, y, w, h)
            p_real = predict_face(face_tensor)  # in [0,1]

            # record the raw frame for optional saving on deny
            recorded_frames.append(frame.copy())
            frame_count += 1

            # vote using probability threshold
            if p_real >= 0.5:   # you can tune to 0.6 later
                real_count += 1
            else:
                fake_count += 1

            # debug line so you can see what's happening
            print(f"[DEBUG] p_real={p_real:.3f} real={real_count} fake={fake_count} frames={frame_count}")

            # stop conditions: enough frames OR time limit
            current_time = time.time()
            if frame_count >= MIN_FRAMES or (start_time and (current_time - start_time) >= MAX_SECONDS):
                break
        else:
            # camera not ready; small pause to avoid busy-waiting
            time.sleep(0.01)

    # make a decision if we actually evaluated anything
    if frame_count > 0:
        evaluate_result(real_count, frame_count, account_number)

    # mark done and release camera
    processing = True
    stop_video_capture()

def process_face(frame, x, y, w, h):
    face = frame[y:y+h, x:x+w]

    face = cv2.fastNlMeansDenoisingColored(face, None, 10, 10, 7, 21)
    
    alpha = 1.5  
    beta = 0    
    face = cv2.convertScaleAbs(face, alpha=alpha, beta=beta)
    
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    face_tensor = transform(face_pil).unsqueeze(0)
    
    return face_tensor

def predict_face(face_tensor):
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = F.softmax(outputs, dim=1)
    return probabilities[0, REAL_CLASS_IDX].item()  # p(real)

def evaluate_result(real_count, frame_count, account_number):
    global result_data, email_sent, recorded_frames

    # Human-friendly time
    current_time_human = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")   # <<< ADDED
    # Filename-safe time
    current_time_fs = datetime.now().strftime("%Y%m%d_%H%M%S")                          # <<< CHANGED (moved here)

    if (real_count / frame_count) >= 0.7:
        result_data['result'] = 'ACCESS GRANTED'
        result_data['color'] = 'green'
        print(f"[{current_time_human}] ✅ ACCESS GRANTED for account {account_number}")  # <<< ADDED
    else:
        result_data['result'] = 'ACCESS DENIED'
        result_data['color'] = 'black'
        print(f"[{current_time_human}] ❌ ACCESS DENIED for account {account_number}")   # <<< ADDED

        video_filename = f"{account_number}_{current_time_fs}.avi"                       # <<< CHANGED (uses new var)

        if not email_sent:
            subject = "⚠️ Liveness Detection Alert"                                      # <<< CHANGED (now inside if)
            body = f"Access denied for account {account_number} at {current_time_human}." # <<< CHANGED (human time)
            send_email(subject, body)                                                    # <<< CHANGED (was print only)
            email_sent = True

        if recorded_frames:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            height, width, layers = recorded_frames[0].shape
            video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

            for frame in recorded_frames:
                video_writer.write(frame)

            video_writer.release()
            print(f"[VIDEO] Saved denial recording as '{video_filename}'")                # <<< ADDED

def gen_frames():
    global camera, result_data, processing

    if camera is None or not camera.isOpened():
        start_video_capture()

    try:
        while True:
            if camera and camera.isOpened():
                success, frame = camera.read()
                if not success:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error in gen_frames: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global camera, email_sent, processing, result_data
    account_number = request.form['account_number']
    session['account_number'] = account_number  
    session['original_url'] = request.referrer  
    email_sent = False
    processing = False
    result_data = {'result': None, 'color': None}
    
    if camera is None or not camera.isOpened():
        start_video_capture()
        threading.Thread(target=capture_frames, args=(account_number,), daemon=True).start()
    
    return redirect(url_for('processing'))

@app.route('/processing')
def processing():
    global result_data
    if result_data['result'] in ['ACCESS GRANTED', 'ACCESS DENIED']:
        return redirect(url_for('final_result'))
    else:
        return render_template('index2.html')

@app.route('/status')
def get_status():
    global result_data
    return jsonify(result_data)

@app.route('/video_capture')
def video_capture():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/final_result')
def final_result():
    result = result_data.get('result', 'Error')
    color = result_data.get('color', 'red')
    original_url = session.get('original_url', url_for('index'))  
    return render_template('stop.html', result=result, color=color, original_url=original_url)

@app.route('/redirect_back', methods=['POST'])
def redirect_back():
    original_url = session.get('original_url', url_for('index'))
    return redirect(original_url)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
