from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/headpose', methods=['POST'])
def headpose():
    try:
        # Execute the Python script
        subprocess.Popen(['python', 'main.py'])
        return jsonify({'message': 'Script execution started.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gaze', methods=['POST'])
def gaze():
    try:
        # Execute the Python script
        subprocess.Popen(['python', 'demo_gaze_estimation.py'])
        return jsonify({'message': 'Script execution started.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500        

if __name__ == '__main__':
    app.run(debug=True)