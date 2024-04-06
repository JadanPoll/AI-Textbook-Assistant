
import asyncio
import http.server
import socketserver
import websockets
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import threading
import json

import signal  # Import the signal module
import socket
import uuid
import logging

# Set the logging level for the requests library to WARNING
logging.getLogger("requests").setLevel(logging.WARNING)

# Set the logging level for the urllib3 library to WARNING
logging.getLogger("urllib3").setLevel(logging.WARNING)

def generate_random_uuid():
    # Generate a random UUID
    random_uuid = uuid.uuid4()

    # Remove hyphens and return the result
    return str(random_uuid).replace('-', '')

def generate_random_uuid2():
            
    # Generate a random UUID
    random_uuid = str(uuid.uuid4())

    # Slice the UUID to create a similar ID
    similar_id = f"aaa{random_uuid[3:]}"
    return similar_id




# Define a class to represent a custom queue
class CustomQueue:
    def __init__(self):
        self.queue = []

    def put(self, parent_id, value, optional=None):
        self.queue.append((parent_id, value, optional))

    def get_by_parent_id(self, parent_id):
        for i, item in enumerate(self.queue):
            if item[0] == parent_id:
                _, value, optional = self.queue.pop(i)
                return [value, optional]
        return [None, None]

    def get(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            return None

    def __len__(self):
        return len(self.queue)

# Example Usage:
queue = CustomQueue()
import http.server
import json

import threading
import time

import http.server
import logging
import http.server


from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from urllib.parse import urlparse, parse_qs


# Disable logging for the http.server module
logging.getLogger('http.server').setLevel(logging.ERROR)

class PollingThread(threading.Thread):
    def __init__(self, request_handler, data_type):
        super().__init__()
        self.request_handler = request_handler
        self.data_type = data_type
        self.stored_data = None


    def run(self):
        stored_data_attr = f"stored_data{self.data_type}"
        stored_data = getattr(self.request_handler, stored_data_attr)


        while self.stored_data is None:

            time.sleep(0.2)  # Sleep for 1 second
            self.stored_data = stored_data = getattr(self.request_handler, stored_data_attr)

        setattr(self.request_handler, stored_data_attr, stored_data)

RESERVE_SERVER=False
class CustomRequestHandler(http.server.SimpleHTTPRequestHandler):

    # Initialize empty dictionaries to store data
    stored_data = {}
    stored_data2 = {}
    stored_data_phind = {}
    stored_data2_phind = {}
    data_lock = threading.Lock()

    # Define a class-level variable to keep track of the count
    get_request_count = 0


    def do_GET(self):
        global httpd
        global server
        global RESERVE_SERVER
        try:
            # Increment the counter each time do_GET is called
            CustomRequestHandler.get_request_count += 1
            print( CustomRequestHandler.get_request_count,"..."*10)
            if self.path == '/':

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()

                # Read HTML content from a file
                html_file_path = 'Socket_Send6.html'
                with open(html_file_path, 'r') as file:
                    html_content = file.read()

                # Send the HTML content to the client
                self.wfile.write(html_content.encode())

                
            elif self.path == '/wake_up':
                RESERVE_SERVER=True
                print(f"..."*100)
                self._set_response(200, 'text/plain')
                self.wfile.write(f"WebSocket server started on ws://localhost:{wss_port}".encode('utf-8'))

            elif self.path == '/get_relay_data':
                if CustomRequestHandler.stored_data:
                    # Get the first key-value pair from stored_data dictionary
                    key, stored_value = next(iter(CustomRequestHandler.stored_data.items()), (None, None))
                    if stored_value:
                        # Delete the entry associated with the key
                        del CustomRequestHandler.stored_data[key]
                        # Encode the key-value pair as JSON
                        response_data = stored_value
                        self._set_response(200, 'application/json')  # Set content type to JSON
                        self.wfile.write(response_data.encode('utf-8'))
                    else:
                        self.send_error_response()
                else:
                    self.send_error_response()



            elif self.path.startswith('/get_relay_data2_phind'):
                # Parse the query parameters to extract the key
                query_params = urlparse(self.path).query
                key = parse_qs(query_params).get('key', [None])[0]
                print("Query Params",key)
                if CustomRequestHandler.stored_data2_phind.get(key,None):
                    # Check if the key exists in the stored_data2_phind dictionary
                    stored_value = CustomRequestHandler.stored_data2_phind.get(key)
                    if stored_value:
                        # Delete the entry associated with the key
                        del CustomRequestHandler.stored_data2_phind[key]
                        self._set_response(200, 'text/plain')
                        self.wfile.write(stored_value.encode('utf-8'))
                    else:
                        self.send_error_response()
                else:
                    self.send_error_response()

            elif self.path.startswith('/get_relay_data2'):
                # Parse the query parameters
                query_params = urlparse(self.path).query
                key = parse_qs(query_params).get('key', [None])[0]
                print(key)
                if CustomRequestHandler.stored_data2.get(key,None):
                    # Check if the key exists in the stored_data2 dictionary
                    stored_value = CustomRequestHandler.stored_data2.get(key)
                    if stored_value:
                        # Delete the entry associated with the key
                        del CustomRequestHandler.stored_data2[key]
                        print("Best Friedddd"*30,stored_value)
                        self._set_response(200, 'text/plain')
                        self.wfile.write(stored_value.encode('utf-8'))
                    else:
                        self.send_error_response()
                else:
                    # Start the polling thread if stored_data2 is empty
                    self.send_error_response()

                    
            elif self.path == '/get_relay_data_phind':
                if CustomRequestHandler.stored_data_phind:
                    # Get the first key-value pair from stored_data dictionary
                    key, stored_value = next(iter(CustomRequestHandler.stored_data_phind.items()), (None, None))
                    if stored_value:
                        # Delete the entry associated with the key
                        del CustomRequestHandler.stored_data_phind[key]
                        # Encode the key-value pair as JSON
                        response_data = stored_value
                        self._set_response(200, 'application/json')  # Set content type to JSON
                        self.wfile.write(response_data.encode('utf-8'))
                    else:
                        self.send_error_response()
                else:

                    self.send_error_response()


            else:
                print(f"Path '{self.path}' not found. Calling superclass method...")
                super().do_GET()
        except KeyboardInterrupt:
            print("Ctrl+C detected. Exiting gracefully.")

            httpd.shutdown()
            server.close()


    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:

                # Process the POST data
                data = post_data.decode('utf-8')

                # Extracting a unique key from POST data
                parsed_data = json.loads(data)
                unique_key = parsed_data.get('unique_key')
                print("SpaceAge"*50,parsed_data,self.path)
                if unique_key:

                    if self.path == '/set_relay_data':
                        CustomRequestHandler.stored_data[unique_key] = data  # Update stored data

                        self._set_response(200, 'text/plain')
                        self.wfile.write("Success: Data received and stored successfully".encode('utf-8'))
                    elif self.path == '/set_relay_data2':
                        CustomRequestHandler.stored_data2[unique_key] = data  # Update stored data

                        self._set_response(200, 'text/plain')
                        self.wfile.write("Success: Data received and stored successfully".encode('utf-8'))


                    elif self.path == '/set_relay_data_phind':
                        CustomRequestHandler.stored_data_phind[unique_key] = data  # Update stored data

                        self._set_response(200, 'text/plain')
                        self.wfile.write("Success: Data received and stored successfully".encode('utf-8'))

                    elif self.path == '/set_relay_data2_phind':
                        print("Storing data from PHIND"*1000,data)
                        CustomRequestHandler.stored_data2_phind[unique_key] = data  # Update stored data

                        self._set_response(200, 'text/plain')
                        self.wfile.write("Success: Data received and stored successfully".encode('utf-8'))

            except Exception as e:
                print("Error processing POST request:", e)
                self._set_response(status_code=400)
                self.wfile.write("Error processing POST request".encode('utf-8'))

        except KeyboardInterrupt:
            print("Ctrl+C detected. Exiting gracefully.")

            httpd.shutdown()
            server.close()







    def _set_response(self, status_code=200, content_type='text/html'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        
    def is_polling_thread_running(self, data_type):
        # Check if a polling thread is already running for the specified data type
        if data_type == "":
            has_polling_thread = hasattr(self, 'polling_thread')
            polling_thread_alive = has_polling_thread and CustomRequestHandler.polling_thread.is_alive()

            return polling_thread_alive
        elif data_type == "2":
            has_polling_thread2 = hasattr(self, 'polling_thread2')
            polling_thread2_alive = has_polling_thread2 and CustomRequestHandler.polling_thread2.is_alive()

            return polling_thread2_alive
        elif data_type == "_phind":
            has_polling_thread_phind = hasattr(self, 'polling_thread_phind')
            polling_thread_alive_phind = has_polling_thread_phind and CustomRequestHandler.polling_thread_phind.is_alive()

            return polling_thread_alive_phind      

        elif data_type == "2_phind":
            has_polling_thread2_phind = hasattr(self, 'polling_thread2_phind')
            polling_thread2_alive_phind = has_polling_thread2_phind and CustomRequestHandler.polling_thread2_phind.is_alive()

            return polling_thread2_alive_phind      

    def send_error_response(self):
        # Send an error response to the client
        self._set_response(400, 'text/plain')
        self.wfile.write("Error: Another request is already being processed. Please try again later.".encode('utf-8'))


# Global variables
connected_clients = set()
connected_clients_lock = asyncio.Lock()  # Add a lock for thread safety

httpd=None


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_http_server(port):
    global httpd
    if is_port_in_use(port):
        print(f"Server is already running on port {port}")
    else:
        print("Nathan Del")
        handler = CustomRequestHandler
        httpd = socketserver.TCPServer(("", port), handler)
        print(f"Serving on port {port}")
        httpd.allow_reuse_address = True
        # Start the HTTP server in a separate thread
        http_thread = threading.Thread(target=httpd.serve_forever)
        http_thread.start()


# Define a coroutine to handle WebSocket connections
async def handle_websocket(websocket, path):
    global connected_clients

    print(f"WebSocket connection established from {websocket.remote_address}"*50)

    connected_clients.add(websocket)
    try:
        while True:
            send_message_command = {
                "type": "send_message",
                "content": f"Hi, who are you",
            }

            # Send the command as a JSON string
            #await websocket.send(json.dumps(send_message_command))

            await asyncio.sleep(1000)  # Change sleep duration

    except websockets.exceptions.ConnectionClosed:
        async with connected_clients_lock:
            connected_clients.remove(websocket)
            print("WebSocket connection closed2"*50)


async def check_server(host, port):
    try:
        # Create a TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Set a timeout for the connection attempt
            s.settimeout(1)
            # Attempt to connect to the server
            s.connect((host, port))
        return True  # Connection successful
    except (socket.timeout, ConnectionRefusedError):
        return False  # Connection failed

host=""
# Define a coroutine to send messages to connected clients
async def send_message_client(message="", user_system_message=None, use_user_system_message=True, create_new_conversation=False,speak=False,stopPlaying=False,
                              use_old_conversation=False,
                              conversation_id=None,send_message_phind=False):
    global connected_clients
    global httpd
    global server
    global host
    global port
    while not await check_server(host, port):
        print("Waiting for server to be active...")
        await asyncio.sleep(1)  # Wait for 1 second before checking again

    while not connected_clients:
        print("Waiting for a client to connect...")
        await asyncio.sleep(1)

    connected_clients_copy = connected_clients.copy()  # separate copy to prevent changes to size while looping
    print(connected_clients_copy)
    for client in reversed(list(connected_clients_copy)):
        # Your code here

        # Your code here
        try:
            parent_id = generate_random_uuid2()

            if create_new_conversation:
                parent_id=None
                create_conversation_command = {
                    "type": "create_new_conversation"
                    #,"request-id": f"{parent_id}"
                }
                await client.send(json.dumps(create_conversation_command))
            elif use_old_conversation:

                parent_id=None
                use_old_conversation = {
                    "type": "use_old_conversation",
                    "conversation_id":conversation_id

                }
                await client.send(json.dumps(use_old_conversation))

            elif use_user_system_message is True and user_system_message is not None:
                parent_id=None
                user_system_command = {
                    "type": "user_system_message",
                    "content": f"{user_system_message}",
                    "bool": use_user_system_message
                }
                await client.send(json.dumps(user_system_command))

            elif speak:
                parent_id=None
                speak_command = {
                    "type": "Speak"
                }
                await client.send(json.dumps(speak_command))
            elif stopPlaying:
                parent_id=None
                stopAudio_command = {
                    "type": "Stop_Audio"
                }
                await client.send(json.dumps(stopAudio_command))
            elif send_message_phind:

                send_message_phind = {
                    "type": "send_message_phind",
                    "content": f"{message}",
                    "parent_id": f"{parent_id}"
                }
                await client.send(json.dumps(send_message_phind))

            else:
                send_message_command = {
                    "type": "send_message",
                    "content": f"{message}",
                    "parent_id": f"{parent_id}"
                }
                await client.send(json.dumps(send_message_command))

            result=[None,None]


            while result[0] is None and not speak and not stopPlaying:
                try:
                    result = queue.get_by_parent_id(parent_id)
                    print("Drusko" * 10, parent_id, result)
                    if result[0]:
                        return result
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("Ctrl+C detected. Exiting gracefully.")
                    connected_clients.remove(client)
                    httpd.shutdown()
                    server.close()
        except websockets.exceptions.ConnectionClosed:
            #connected_clients.remove(client)
            #server.close()
            pass



# Define a coroutine to receive messages from connected clients

async def receive_messages():
    global httpd
    global connected_clients
    global queue
    global server
    try:
            
        async with connected_clients_lock:
            clients = list(connected_clients)
            print("Nathan2"*10,clients)
            for client in reversed(clients):
                try:
                    try:
                        # Wait for client.recv() with a timeout of 0.1 seconds (adjust as needed)
                        message = await asyncio.wait_for(client.recv(), timeout=5)


                    except asyncio.TimeoutError:
                        print("ISKIP iskip")
                        # Handle the case where no message is received within the timeout
                        continue


                    if message.startswith("pdata: "):
                        data = json.loads(message.replace("pdata:", "").strip())

                        queue.put(data["message_id"],data["text"],[data["WebResults"]])
                        return
                    elif message.startswith("data: "):
                        data = json.loads(message.replace("data:", "").strip())

                        parts_zero = data["message"]["content"]["parts"][0]
                        queue.put(data["message"]["metadata"]["parent_id"],parts_zero,[data["conversation_id"] ,data['message']['id']])
                        return
                    elif message.startswith("createFile: "):
                        data = json.loads(message.replace("createFile:", "").strip())

                        # Create/write to file the JSON data
                        with open("CHAToutput.json", "w") as f:
                            json.dump(data, f)
                        return
                    else:
                        queue.put(None,message)
                        return

                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed0"*50)
                    connected_clients.remove(client)
                    #httpd.shutdown()
                    #server.close()

    except KeyboardInterrupt:
        httpd.shutdown()
        server.close()

# Define the main coroutine
server=""

port = 8000
wss_port = 8767
def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True  # Connection successful, port is open
    except (socket.timeout, ConnectionRefusedError):
        return False  # Connection failed, port is not open or not reachable



async def restart_websocket_server():
    global RESERVE_SERVER
    global server

async def start_server():
    while True:
        print("Restart Server Code Running")

        if RESERVE_SERVER:
            try:
                server = await websockets.serve(handle_websocket, "localhost", wss_port)
                print(f"WebSocket server started on ws://localhost:{wss_port}")
                RESERVE_SERVER = False
            except OSError as e:
                print(f"Error: {e}. WebSocket server is already running.")
        await asyncio.sleep(5)  # Adjust the sleep duration as needed



async def main():
    global httpd
    global host
    host = "localhost"

    ws_url = f"ws://{host}:{wss_port}"
    global server
    if is_port_open(host, wss_port):
        print(f"WebSocket server is already listening on {ws_url}")
        return
    else:
        print(f"No WebSocket server found on {ws_url}")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--start-minimized') # This argument is supported by Chrome as well

    # Initialize the Chrome driver with the specified options
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    # Navigate to the specified URL
    driver.get(f"http://localhost:{port}")


    """

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--start-minimized')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(f"http://localhost:{port}")
    """


    server = await websockets.serve(handle_websocket, "localhost", wss_port)
    print(f"WebSocket server started on ws://localhost:{wss_port}")
    global RESERVE_SERVER
    # Start the WebSocket server in a background task
    # Run the setup_chrome_driver function in a separate thread
    # Other parts of your code...
    asyncio.create_task(restart_websocket_server())


    try:
        while True:
            await receive_messages()

            await asyncio.sleep(1)
    except KeyboardInterrupt:
        httpd.shutdown()
        pass
    finally:
        server.close()
        await server.wait_closed()

print("Nathan Not here")
# Start the HTTP server
start_http_server(port)
time.sleep(3)
# Start the WebSocket server in a separate thread using asyncio.run_coroutine_threadsafe
loop=""
def start_websocket_server():
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

websocket_thread = threading.Thread(target=start_websocket_server)
websocket_thread.start()

import time


# Define the signal handler for SIGINT
def signal_handler(sig, frame):
    global server
    global httpd
    print("Received SIGINT. Closing WebSocket server...")
    loop.stop()  # Stop the event loop
    httpd.shutdown()
    server.close()  # Close the WebSocket server
    # You can perform additional cleanup steps here if needed
    print("WebSocket server closed. Exiting...")
    exit()
# Set up the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)
