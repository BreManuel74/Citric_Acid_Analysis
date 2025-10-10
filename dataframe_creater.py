import pandas as pd
import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def combine_csv_files(directory_path=None, output_filename="master_data.csv"):
    """
    Combine multiple CSV files into one master CSV file.
    
    Parameters:
    directory_path (str): Path to directory containing CSV files. If None, uses current directory.
    output_filename (str): Name for the output master CSV file.
    
    Returns:
    pandas.DataFrame: Combined dataframe
    """
    
    # If no directory specified, use current directory
    if directory_path is None:
        directory_path = os.getcwd()
    
    # Create pattern to find all CSV files
    csv_pattern = os.path.join(directory_path, "*.csv")
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return None
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # List to store individual dataframes
    dataframes = []
    
    # Process each CSV file
    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            dataframes.append(df)
            print(f"Successfully read {file_path}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            continue
    
    if not dataframes:
        print("No valid CSV files could be read.")
        return None
    
    # Combine all dataframes
    master_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the master dataframe
    output_path = os.path.join(directory_path, output_filename)
    master_df.to_csv(output_path, index=False)
    
    print(f"\nMaster CSV created: {output_path}")
    print(f"Total rows: {len(master_df)}")
    print(f"Total columns: {len(master_df.columns)}")
    print(f"Columns: {list(master_df.columns)}")
    
    return master_df


def combine_selected_csv_files(initial_dir=None):
    """
    Open a file picker to select specific CSV files and combine them.

    Parameters:
    initial_dir (str): Initial directory for the file dialog. Defaults to current directory.

    Returns:
    pandas.DataFrame | None: Combined dataframe if files selected and read; otherwise None.
    """

    # Initialize Tkinter root (hidden) so dialogs appear without a full window
    root = tk.Tk()
    root.withdraw()
    try:
        # Bring the dialog to the front
        root.attributes('-topmost', True)
    except Exception:
        # Fallback if attributes isn't supported in some environments
        pass

    if initial_dir is None:
        initial_dir = os.getcwd()

    # Ask user to select one or more CSV files
    file_paths = filedialog.askopenfilenames(
        title="Select CSV files to combine",
        initialdir=initial_dir,
        filetypes=[("CSV Files", "*.csv")]
    )

    # User cancelled
    if not file_paths:
        print("No files selected. Operation cancelled.")
        return None

    print(f"Selected {len(file_paths)} CSV files:")
    for p in file_paths:
        print(f"  - {os.path.basename(p)}")

    dataframes = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"Successfully read {file_path}: {len(df)} rows")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")

    if not dataframes:
        print("No valid CSV files could be read.")
        return None

    master_df = pd.concat(dataframes, ignore_index=True)

    # Ask where to save the combined CSV
    default_dir = os.path.dirname(file_paths[0]) if file_paths else initial_dir
    save_path = filedialog.asksaveasfilename(
        title="Save combined CSV as...",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")],
        initialdir=default_dir,
        initialfile="master_data.csv",
    )

    if save_path:
        try:
            master_df.to_csv(save_path, index=False)
            print(f"\nMaster CSV created: {save_path}")
            print(f"Total rows: {len(master_df)}")
            print(f"Total columns: {len(master_df.columns)}")
            print(f"Columns: {list(master_df.columns)}")
            try:
                messagebox.showinfo("Success", f"Combined CSV saved to:\n{save_path}")
            except Exception:
                # messagebox may fail in some headless contexts; ignore
                pass
        except Exception as e:
            print(f"Failed to save combined CSV: {e}")
            try:
                messagebox.showerror("Save Failed", str(e))
            except Exception:
                pass
    else:
        print("Save cancelled. Returning combined DataFrame without saving.")

    return master_df

# Example usage
if __name__ == "__main__":
    # Option A: Use a GUI to select specific CSV files to combine
    master_data = combine_selected_csv_files()
    
    # Option B: Use current directory (combine all CSVs in a folder)
    # master_data = combine_csv_files()

    # Option C: Specify a different directory
    # master_data = combine_csv_files(r"C:\path\to\your\csv\files")
    
    # Option D: Specify directory and custom output filename
    # master_data = combine_csv_files(r"C:\path\to\your\csv\files", "combined_mouse_data.csv")
    
    # Display first few rows of the combined data
    if master_data is not None:
        print("\nFirst 5 rows of combined data:")
        print(master_data.head())
        
        # Only compute ID stats if column exists
        if 'ID' in master_data.columns:
            print(f"\nUnique mice in dataset: {master_data['ID'].nunique()}")
            print(f"Mouse IDs: {sorted(master_data['ID'].unique())}")
        else:
            print("\nColumn 'ID' not found in combined data; skipping ID summary.")

