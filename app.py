import os
import cv2
import time
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from pose_engine import DrillPoseEngine
from pdf_generator import generate_drill_pdf

os.makedirs(os.path.join("static", "results"), exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

is_tracking = False
captured_session_buffer = []
engine = DrillPoseEngine()

# SECUREEYE_RTSP_URL = "rtsp://admin:password@192.168.231.99:554/stream1"

def secureeye_camera_worker(drill_type):
    global is_tracking, captured_session_buffer
    print("[CAMERA] Initializing Camo Studio Virtual Driver...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("[CAMERA] Index 1 unavailable. Falling back to Index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[CAMERA] CRITICAL ERROR: Could not open any local camera devices.")
        is_tracking = False
        return   

    while is_tracking:
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.01)
            continue
            
        annotated_img, eval_metrics = engine.evaluate_drill_frame(frame, drill_type)
        if eval_metrics:
            payload = {'metrics': eval_metrics, 'drill_type': drill_type}
            # Thread-safe appending
            if is_tracking:
                captured_session_buffer.append({'frame': annotated_img, 'metrics': eval_metrics, 'drill_type': drill_type})
                socketio.emit('live_telemetry', payload)
            
        time.sleep(0.01)
        
    cap.release()

# --- WEB ROUTING LAYOUTS ---
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/tv_display')
def tv_display(): 
    return render_template('tv.html')

@app.route('/<drill_type>')
def control_panel(drill_type): 
    return render_template('control.html', drill_type=drill_type)

# --- SYSTEM CONTROLLERS ---
@app.route('/api/start', methods=['POST'])
def start_tracking():
    global is_tracking, captured_session_buffer
    drill_type = request.json.get('drill_type', 'salute')
    
    if not is_tracking:
        # FIX: Explicitly clear old run history out of memory on initialization
        captured_session_buffer = [] 
        engine.history_buffer = [] # Reset the YOLO frame smoothing buffer too
        is_tracking = True
        socketio.start_background_task(secureeye_camera_worker, drill_type)
        return jsonify({"status": "success", "message": "Tracking pipeline active."})
    return jsonify({"status": "error", "message": "Engine busy."})

@app.route('/api/stop', methods=['POST'])
def stop_tracking():
    global is_tracking, captured_session_buffer
    if is_tracking:
        is_tracking = False
        
        # FIX: Forced deterministic thread sleep to guarantee the camera releases 
        # and appends stop completely before reading data matrix structures.
        time.sleep(0.5) 
        
        if captured_session_buffer:
            snapshot_path = os.path.join("static", "results", "evidence_snapshot.jpg")
            pdf_path = os.path.join("static", "results", "latest_report.pdf")
            
            error_frames = [d for d in captured_session_buffer if d['metrics']['status'] in ['Fail', 'Warning']]
            worst_node = error_frames[0] if error_frames else captured_session_buffer[len(captured_session_buffer)//2]
            
            cv2.imwrite(snapshot_path, worst_node['frame'])
            generate_drill_pdf(captured_session_buffer, snapshot_path, pdf_path)
            
            socketio.emit('new_pdf_ready', {'pdf_url': '/static/results/latest_report.pdf'})
            return jsonify({"status": "success", "message": "Evaluation finalized cleanly."})
            
        return jsonify({"status": "error", "message": "No frame data compiled during test windows."})
    return jsonify({"status": "error", "message": "Tracking loop was not active."})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)