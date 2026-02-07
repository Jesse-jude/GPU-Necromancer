from huggingface_hub import hf_hub_download
import os

os.makedirs('models', exist_ok=True)

print(" Downloading Mistral 7B (4.4GB)...")
model_path = hf_hub_download(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
    filename='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    cache_dir='./models'
)
print(f" Downloaded: {model_path}")