from flask import Flask, request
from flask_socketio import SocketIO
import base64
import io
from PIL import Image
import threading
import time
from flask_cors import CORS
import imagedata_pb2
# from ..localizev2 import query_processing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_query_image import query_processing

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

def send_hello():
    while True:
        socketio.emit('message', {'data': 'hello'})
        time.sleep(1)

@app.route('/')
def index():
    return 'WebSocket server is running!'

@socketio.on('connect')
def handle_connect(sid):
    print(f'Client connected, SID: {sid}')
    threading.Thread(target=send_hello).start()


@socketio.on('sendProtobufImage')
def handle_protobuf_image(data):
    try:
        image_message = imagedata_pb2.ImageMessage()
        image_message.ParseFromString(data)
        image_data = image_message.image_data
        image_format = image_message.image_format.lower()

        dataset_name = 'testtrack'
        base_dir = Path(__file__).resolve().parent.parent  # Adjust as needed
        image_save_path = base_dir / f'datasets/{dataset_name}/path_queries/query.{image_format}'
        image_save_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(io.BytesIO(image_data))
        
        image.save(str(image_save_path), image_format.upper())

        relative_positions = query_processing(dataset_name, str(image_save_path))

        if relative_positions:
            socketio.emit('localizationResults', {'data': relative_positions})
        else:
            socketio.emit('imageResponse', {'data': 'Localization failed'})

    except Exception as e:
        print(f"Error processing Protobuf image: {e}")
        socketio.emit('imageResponse', {'data': f'Error: {e}'})


@socketio.on('testMessage')
def handle_test_message(data):
    print(f'Received test message: {data}')


if __name__ == '__main__':
    def start_server():
        socketio.run(app, debug=False, host='0.0.0.0', port=12345)

    start_server()
