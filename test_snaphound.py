import sys
import json
import threading
import os
from snaphoundpy import SnapHound

# Handle optional paths argument
input_data = json.loads(sys.argv[1])

print("Search Value:", input_data.get("search_data", ""))
print("Priority Paths:", input_data.get("priority_paths", []))
print("Paths:", input_data.get("path", []))
print("Index:", input_data.get("index", False))

# Initialize SnapHound
snaphoundpy = SnapHound(paths=input_data.get("path", []), priority_paths=input_data.get("priority_paths", []))
print("SnapHound Started.")
# Global variable to keep track of the indexing thread
index_thread = None
index_lock = threading.Lock()

# Function to index images in a separate thread
def index_images_thread():
	global index_thread
	snaphoundpy.index_images()
	index_thread = None  # Reset thread reference when done

# Function to start indexing if not already running
def start_indexing():
	global index_thread
	with index_lock:
		if index_thread is None or not index_thread.is_alive():
			index_thread = threading.Thread(target=index_images_thread, daemon=True)
			index_thread.start()

def search(search_data):
	if search_data:
		if os.path.isfile(search_data):
			result = snaphoundpy.search_with_image(search_data)
		else:
			result = snaphoundpy.search_with_text(search_data)
		print("Search result:", result)
	return True

def main(data):
	input_data["search_data"] = data.get("search_data", "")
	input_data["index"] = data.get("index", False)

	if input_data["index"]:
		start_indexing()

	if input_data["search_data"]:
		search(input_data["search_data"])

def server_mode():
	main(input_data)
	while True:
		try:
			user_input = sys.stdin.readline().strip()
			if not user_input:
				continue

			args = json.loads(user_input)
			main(args)

			sys.stdout.flush()

		except (EOFError, KeyboardInterrupt):
			break

# Start server mode
server_mode()