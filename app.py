import sys
from requirements_utils import find_missing

def check_requirements():
    """Check missing packages and exit cleanly if needed."""
    missing = find_missing("requirements.txt")
    if missing:
        print("\n❌ Missing required packages:")
        for pkg in missing:
            print(" -", pkg)
        print("\nGo to Terminal(alt+F12) than Run:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("✔️ All requirements are satisfied!")
    return True

if __name__ == "__main__":
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)

    print("\n🚀 Launching application...")
