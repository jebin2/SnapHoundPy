from setuptools import setup, find_packages

setup(
	name="snaphound",
	version="0.1.0",
	packages=find_packages(exclude=["tests*"]),
	install_requires=[
		"sentencepiece==0.2.0",
		"python-dotenv==1.0.1",
		"transformers==4.50.0.dev0",
		"torch==2.6.0",
		"pillow==11.1.0",
		"numpy==2.2.3",
		"faiss-cpu==1.10.0",
		"protobuf==5.29.3"
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
	python_requires=">=3.8",
)