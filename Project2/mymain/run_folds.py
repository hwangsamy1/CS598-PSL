# https://campuswire.com/c/GB46E5679/feed/822
# Import necessary modules
import os  # Provides functions to interact with the operating system (e.g., change directories, run system commands)
import time  # Provides time-related functions (e.g., track elapsed time)
from tqdm import tqdm  # A module for showing progress bars in loops (though not used in the code snippet provided)


# Define the function `run_fold` that takes two arguments:
# - `fold_data`: the directory where the code will run
# - `code_path`: the relative or absolute path to the Python script to be executed
def run_fold(fold_data, code_path):
    # Change the current working directory to the specified folder (fold_data)
    os.chdir(fold_data)

    # Print a message to indicate that the execution of the code is starting
    print(f"Start Running {code_path}: in {fold_data}")

    # Record the starting time for the execution
    start_time = time.time()

    # Attempt to run the script at `code_path` using the system's shell (python3 command)
    try:
        # Run the python script located at `code_path` using the system's Python interpreter
        os.system(f"python3 {code_path}")

        # Print a success message if the script ran without issues
        print(f"Successfully executed {code_path} in {fold_data}")

    # Handle any exceptions that might occur during the execution of the script
    except Exception as e:
        # Print an error message if the script fails to run (e.g., file not found, script error)
        print(f"Error running {code_path} in {fold_data}: {e}")

    # Record the ending time after the script has finished executing
    end_time = time.time()

    # Calculate the total execution time by subtracting the start time from the end time
    execution_time = end_time - start_time

    # Print the execution time, formatted to 2 decimal places, indicating how long the script took to execute
    print(f"Execution time for {fold_data}: {execution_time:.2f} seconds")


# This is the main entry point for the script, which will only execute when the script is run directly (not imported as a module).
if __name__ == "__main__":
    # Create a list of strings 'fold_1' to 'fold_10' using a list comprehension.
    # This list represents different directories (folds) where data is stored for processing.
    folds = [f"fold_{i}" for i in range(1, 11)]  # ['fold_1', 'fold_2', ..., 'fold_10']

    # Get the current working directory (cwd), where the script is being run from.
    cwd = os.getcwd()

    # Print the current working directory to the console.
    print(f"cwd: {cwd}")

    # Construct the full path to the project data directory by joining the current working directory (cwd) with the folder 'Proj2_Data'.
    # The `os.path.join()` function ensures the path is correctly formed for the current operating system (handles slashes correctly).
    project_data = os.path.join(cwd, 'Proj2_Data')

    # Print the full path of the project data directory to the console.
    print(f"Project Data: {project_data}")

    # Construct the full path to the script 'mymain.py' by joining the current working directory (cwd) with the script name.
    # This will be the path where the script 'mymain.py' is located.
    code_path = os.path.join(cwd, 'mymain.py')

    # Print the path to the 'mymain.py' script to the console.
    print(f"code path: {code_path}")

    # Loop through each fold (from 'fold_1' to 'fold_10'), using tqdm to show a progress bar.
    # tqdm is a Python library that adds a progress bar to any iterable. Here, it's used to show progress while iterating over the folds.
    # The `desc` parameter adds a description that appears before the progress bar.
    for fold in tqdm(folds, desc='Running Project 2'):
        # For each fold, construct the full path to the corresponding folder within the 'Proj2_Data' directory.
        fold_data = os.path.join(project_data, fold)

        # Call the previously defined `run_fold` function, passing the fold data and code path as arguments.
        # This function will change the directory to the fold, execute 'mymain.py' in that context, and report execution time.
        run_fold(fold_data, code_path)

        # Print an empty line for better readability of the output between fold executions.
        print()

    # Once all the folds have been processed, print a message indicating that all folds have been handled.
    print("All folds processed!")