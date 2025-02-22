from setuptools import setup, find_packages

setup(
	name="snaphound",
	version="0.1.0",
	packages=find_packages(exclude=["tests*"]),
	install_requires=[
		"SentencePiece",
		"pymediainfo",
		"python-dotenv",
		"transformers",
		"torch",
		"pillow",
		"numpy",
		"faiss-cpu",
		"protobuf"
	],
	author="Jebin Einstein",
	description="Search through images and videos using AI",
	long_description=open("README.md").read(),
	long_description_content_type="text/markdown",
	url="https://github.com/jebin2/SnapHoundPy",
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	python_requires=">=3.7",
)