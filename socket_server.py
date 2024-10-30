import socket
import threading
from chatbot_app import process_query  # Ensure you import process_query from chatbot_app

# Server configuration
HOST = '127.0.0.1'
PORT = 65432

def handle_client(conn, addr):
    print(f"Connected by {addr}")
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            query = data.decode("utf-8")
            result = process_query(query)
            conn.sendall(result.encode("utf-8"))
        except Exception as e:
            print(f"Error processing request from {addr}: {e}")
            break
    conn.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    print(f"Server listening on {HOST}:{PORT}")
    
    while True:
        try:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()
        except Exception as e:
            print(f"Error accepting connection: {e}")

if __name__ == "__main__":
    start_server()
