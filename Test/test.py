import os

# The specific folders from your architecture
folders = [
    "input", 
    "brain", 
    "memory", 
    "engine", 
    "output", 
    "onboarding", 
    "teacher"
]

def add_inits():
    for folder in folders:
        if os.path.isdir(folder):
            init_file = os.path.join(folder, "__init__.py")
            # Create a blank file if it doesn't exist
            with open(init_file, "a") as f:
                pass 
            print(f"✅ Ensured __init__.py in: {folder}/")
        else:
            print(f"⚠️  Skipping: {folder}/ (folder not found)")

if __name__ == "__main__":
    add_inits()