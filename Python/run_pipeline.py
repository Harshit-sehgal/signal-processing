import subprocess
import os
import sys

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n==================================================")
    print(f"🚀 RUNNING STAGE: {script_name}")
    print(f"==================================================")
    
    # Run using the current python executable (the virtual env python)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"❌ Error occurred in {script_name}. Aborting pipeline.")
        sys.exit(result.returncode)
    else:
        print(f"✅ Finished STAGE: {script_name} successfully!")

if __name__ == "__main__":
    print("==================================================")
    print("🌟 PG-AMCD Pipeline Master Execution Runner 🌟")
    print("==================================================")
    
    # Stage 1: Adaptive Preprocessing & CEEMDAN Decomposition
    run_script("iceemdan.py")
    
    # Stage 2: Multi-Criteria Adaptive IMF Weighting (MAIW)
    run_script("maiw_weighting.py")
    
    # Stage 3: Bayesian Adaptive Wavelet Denoising
    run_script("wavelet_denoise.py")
    
    print("\n==================================================")
    print("🎉 ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("==================================================")
