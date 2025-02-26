import sys
import json
import threading
import os
from snaphoundpy import SnapHound

# Handle optional paths argument
paths = json.loads(sys.argv[1])

print("Search Value:", paths.get("search_data", ""))
print("Priority Paths:", paths.get("priority_paths", []))
print("Paths:", paths.get("path", []))
print("Index:", paths.get("index", False))

# Initialize SnapHound
snaphoundpy = SnapHound(paths=paths.get("path", []), priority_paths=paths.get("priority_paths", []))
print("SnapHound Started.")
# Global variable to keep track of the indexing thread
index_thread = None
index_lock = threading.Lock()

# Function to index images in a separate thread
def index_images_thread():
	global index_thread
	print("Starting image indexing...")
	snaphoundpy.index_images()
	print("Indexing completed.")
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

def server_mode():
	while True:
		try:
			user_input = sys.stdin.readline().strip()
			if not user_input:
				continue

			args = json.loads(user_input)
			paths["search_data"] = args.get("search_data", "")
			paths["index"] = args.get("index", False)

			if paths["index"]:
				start_indexing()

			if paths["search_data"]:
				search(paths["search_data"])

			sys.stdout.flush()

		except (EOFError, KeyboardInterrupt):
			break

# Start server mode
server_mode()
