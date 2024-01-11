import websocket
import localize_pb2  # Import your Protobuf schema module

def create_test_message():
    # Create an instance of your Protobuf message
    message = localize_pb2.LocalizationData()
    # Set the fields of your message
    message.x = 1.0
    message.y = 1.0
    message.z = 1.0
    # More fields as per your Protobuf schema
    return message

def send_message(message):
    # Serialize the Protobuf message to a binary format
    serialized_message = message.SerializeToString()

    # Create a WebSocket connection to the server
    ws = websocket.create_connection("ws://0.0.0.0:42069")
    # Replace YOUR_SERVER_PORT with the actual port number

    # Send the serialized message
    ws.send(serialized_message, opcode=websocket.ABNF.OPCODE_BINARY)

    # Receive server response
    response = ws.recv()
    print("Received:", response)

    # Close the connection
    ws.close()

if __name__ == "__main__":
    test_message = create_test_message()
    send_message(test_message)
