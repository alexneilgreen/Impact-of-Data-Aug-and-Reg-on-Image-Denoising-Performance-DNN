import os
import subprocess
import sys
import time

def run_script(script_name):
    """
    Run a Python script in the PreProcess folder
    
    Args:
        script_name (str): Name of the script to run
    """
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}")
    
    script_path = os.path.join("PreProcess", script_name)
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"ERROR: Script {script_path} not found!")
        return False
    
    # Set the working directory to the PreProcess folder
    try:
        process = subprocess.run([sys.executable, script_name], 
                                check=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                text=True,
                                cwd="PreProcess")  # Run from PreProcess directory
        
        # Print script output
        print(process.stdout)
        
        if process.stderr:
            print("ERRORS:")
            print(process.stderr)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_name}:")
        print(e.stderr)
        return False

def main():
    """Main function to run all pre-process analysis scripts"""
    start_time = time.time()
    
    # List of scripts to run
    scripts = [
        "L1PreProcessAnalyze.py",
        "L2PreProcessAnalyze.py",
        "ESPreProcessAnalyze.py",
        "DRPreProcessAnalyze.py"
    ]
    
    success_count = 0
    
    # Run each script
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Analysis complete! {success_count}/{len(scripts)} scripts executed successfully")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()