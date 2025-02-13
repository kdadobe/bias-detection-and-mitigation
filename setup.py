from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="bias_detection",  # Package name
    version="0.1.0",  # Version number
    author="Kinjal P Darji",
    author_email="kdadobe1@gmail.com",
    description="A toolkit to detect and mitigate bias in text using a BERT model",
    long_description=open("README.md").read(),  # Read from README.md
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically find package directories
    install_requires=requirements,  # Dependencies are loaded from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "bias-detect=api.main:app",  # Allows CLI execution
        ]
    },
    include_package_data=True,  # Ensures non-code files are included
)
