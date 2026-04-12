import sys
import subprocess
import signal
from flask import Flask, render_template, jsonify, send_from_directory
import os

app = Flask(__name__)

# Track the running camera process
camera_process = None

@app.route('/')
def home():
    return render_template('services.html')

@app.route('/<path:filename>')
def serve_file(filename):
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return "File not found", 404


@app.route('/run_python_program')
def run_python_program():
    global camera_process

    # If already running, don't start another
    if camera_process is not None and camera_process.poll() is None:
        return jsonify({"status": "already_running", "message": "Camera is already running."})

    try:
        # Launch as a non-blocking background process so Flask doesn't hang
        camera_process = subprocess.Popen(
            [sys.executable, 'sample.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return jsonify({"status": "started", "message": "Camera started successfully.", "pid": camera_process.pid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop_python_program')
def stop_python_program():
    global camera_process

    if camera_process is None or camera_process.poll() is not None:
        return jsonify({"status": "not_running", "message": "No camera process is running."})

    try:
        camera_process.terminate()
        camera_process.wait(timeout=5)
        camera_process = None
        return jsonify({"status": "stopped", "message": "Camera stopped successfully."})
    except subprocess.TimeoutExpired:
        camera_process.kill()
        camera_process = None
        return jsonify({"status": "killed", "message": "Camera process force-killed."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
