from transformers import AutoProcessor, AutoModel
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
model_name = "google/siglip-so400m-patch14-384"
save_path = f"{PACKAGE_DIR}/model_files/"

def __download_models():
	print(f"Downloading model: {model_name}")

	# Download and save the model
	model = AutoModel.from_pretrained(model_name)
	model.save_pretrained(save_path)
	print(f"Model saved at: {save_path}")

def load_model():
	__download_models()
	print(f"Using model_name: {model_name}")
	processor = AutoProcessor.from_pretrained(model_name)
	processor.save_pretrained(save_path)
	print(f"Processor saved at: {save_path}")

	# Now, load from disk (no internet needed)
	print("Loading model from disk...")
	model = AutoModel.from_pretrained(save_path).eval()
	print("Model loaded successfully.")

	print("Loading processor from disk...")
	processor = AutoProcessor.from_pretrained(save_path)
	print("Processor loaded successfully.")

	return model, processor