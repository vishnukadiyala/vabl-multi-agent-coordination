
import os
import sys
import shutil
import platform
import zipfile
import urllib.request
import time
from pathlib import Path

def setup_smac():
    print("="*60)
    print("Setting up StarCraft II and SMAC Maps")
    print("="*60)

    # 1. Determine SC2 Path
    sc2_path = os.environ.get("SC2PATH")
    if not sc2_path:
        if platform.system() == "Windows":
            sc2_path = r"C:\Program Files (x86)\StarCraft II"
        elif platform.system() == "Darwin":
            sc2_path = "/Applications/StarCraft II"
        else: # Linux
            sc2_path = os.path.expanduser("~/StarCraftII")
        
        print(f"SC2PATH not set. Defaulting to: {sc2_path}")
        os.environ["SC2PATH"] = sc2_path
    else:
        print(f"Using defined SC2PATH: {sc2_path}")

    sc2_path = Path(sc2_path)
    if not sc2_path.exists():
        print(f"ERROR: StarCraft II installation not found at {sc2_path}")
        print("Please install StarCraft II or set SC2PATH to the correct location.")
        return False

    # 2. Check/Install Maps
    maps_dir = sc2_path / "Maps"
    smac_maps_dir = maps_dir / "SMAC_Maps"
    
    if smac_maps_dir.exists():
        print(f"SMAC Maps found at: {smac_maps_dir}")
    else:
        print(f"SMAC Maps not found. Attempting installation...")
        
        # Check write permissions
        try:
            if not maps_dir.exists():
                maps_dir.mkdir(parents=True, exist_ok=True)
            test_file = maps_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            print(f"ERROR: No write permission for {maps_dir}")
            print("Cannot install maps automatically in this location.")
            print("Please run as Administrator or manually download maps.")
            return False

        map_url = "https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip"
        zip_path = maps_dir / "SMAC_Maps.zip"
        
        print(f"Downloading maps from {map_url}...")
        try:
            urllib.request.urlretrieve(map_url, zip_path)
            print("Download complete. Extracting...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(maps_dir)
            
            print("Extraction complete.")
            if zip_path.exists():
                os.remove(zip_path)
                
        except Exception as e:
            print(f"ERROR during map installation: {e}")
            return False

    # 3. Verify Environment
    print("\nVerifying SMAC environment...")
    try:
        from smac.env import StarCraft2Env
        
        # Create a simple environment (3m = 3 Marines vs 3 Marines)
        env = StarCraft2Env(map_name="3m", difficulty="1")
        env_info = env.get_env_info()
        
        print("Successfully initialized StarCraft2Env!")
        print(f"Environment Info: {env_info}")
        
        env.reset()
        print("Reset successful.")
        
        env.close()
        print("Environment verification passed.")
        return True
        
    except ImportError:
        print("ERROR: 'smac' or 'pysc2' package not found.")
        print("Please install with: pip install pysc2 smac")
        return False
    except Exception as e:
        print(f"ERROR during verification: {e}")
        return False

if __name__ == "__main__":
    success = setup_smac()
    sys.exit(0 if success else 1)
