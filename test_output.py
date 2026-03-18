import sys
print("TEST START")
sys.stdout.flush()
try:
    import torch
    print(f"Torch Version: {torch.__version__}")
except Exception as e:
    print(f"Torch Error: {e}")
sys.stdout.flush()
print("TEST END")
