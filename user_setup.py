# This file is used to setup the project. It is executed when the project is imported.
# This file should be used to download all large files (e.g., model weights) and store them to disk.
# In this file, you can also check if the environment works as expected.
# If something goes wrong, you can exit the script with a non-zero exit code.
# This will help you detect issues early on.

import subprocess
import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def install_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
        return True
    except subprocess.CalledProcessError:
        return False


def download_large_files():
    try:
        model_name = "Helsinki-NLP/opus-mt-mul-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained("model")
        tokenizer.save_pretrained("model")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def check_packages():
    required_packages = ["datasets", "transformers", "evaluate", "torch", "sentencepiece"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Package {package} is not installed.")
            return False
    return True


def check_environment():
    try:
        import transformers
        import torch
        return check_packages()
    except ImportError as e:
        print(f"Import error: {e}")
        return False


if __name__ == "__main__":
    print("Perform your setup here.")

    if not check_environment():
        print("Environment check failed.")
        if not install_packages():
            print("Package installation failed.")
            exit(1)

    if not download_large_files():
        print("Downloading large files failed.")
        exit(1)

    print("Setup completed successfully. The environment and packages are installed.")
    exit(0)
