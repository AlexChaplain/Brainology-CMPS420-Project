import tkinter as tk
from tkinter import filedialog
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# function to display EEG data in the first pane
def display_eeg_data(file_path):
    global first_pane_label
    global first_pane_figure
    first_pane_label.config(text="EEG Data")

    try:
# Load  EEG data from the selected .edf file
        raw = read_raw_edf(file_path, preload=True)

# Plot the EEG data in the first pane with scaling
        fig = raw.plot(scalings={"eeg": 1e-5}, show=False)  # Adjust scaling as needed
        first_pane_figure = fig

# Update the first pane with the EEG plot
        canvas = FigureCanvasTkAgg(fig, master=first_pane)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        print(f"Error loading or plotting EEG data: {e}")
        
# main window
root = tk.Tk()
root.title("EEG Data Analysis")

# styles
style = {
    "body": {"font-family": "Arial, sans-serif", "margin": 0, "padding": 0, "height": "100vh"},
    "header": {"background-color": "#333", "color": "white", "text-align": "center", "padding": "10px"},
    "container": {"flex": 1, "display": "flex", "width": "100%"},
    "pane": {"flex": 1, "border": "1px solid #ccc", "border-radius": "5px", "background-color": "#f9f9f9", "box-shadow": "2px 2px 5px rgba(0, 0, 0, 0.2)", "padding": "20px"},
    "file-upload": {"margin-bottom": "10px"},
    "analysis-results": {"white-space": "pre-wrap"},
}

# header
header = tk.Frame(root, bg=style["header"]["background-color"])
header.pack()
header_label = tk.Label(header, text="Brainology EEG Data Analysis", font=("Arial", 16), bg=style["header"]["background-color"], fg=style["header"]["color"])
header_label.pack()

# file upload/analyze button
file_upload_frame = tk.Frame(root)
file_upload_frame.pack()
file_upload_label = tk.Label(file_upload_frame, text="Upload EEG Data:", font=("Arial", 12))
file_upload_label.pack()
file_upload_button = tk.Button(file_upload_frame, text="Browse", command=lambda: browse_file())
file_upload_button.pack()
analysis_results_label = tk.Label(file_upload_frame, text="", font=("Arial", 12), wraplength=400, justify="left")
analysis_results_label.pack()

# Container for panes
container = tk.Frame(root)  
container.pack(fill=tk.BOTH, expand=True)

# Pane  borders
pane_border_color = "#ccc"  
pane_border_width = 1  

# first pane
first_pane = tk.Frame(container, bg=style["pane"]["background-color"], borderwidth=pane_border_width, relief="solid", highlightbackground=pane_border_color, highlightthickness=pane_border_width)
first_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
first_pane_label = tk.Label(first_pane, text="First Pane", font=("Arial", 14))
first_pane_label.pack()
first_pane.pack_propagate(False)

# second pane
second_pane = tk.Frame(container, bg=style["pane"]["background-color"], borderwidth=pane_border_width, relief="solid", highlightbackground=pane_border_color, highlightthickness=pane_border_width)
second_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
second_pane_label = tk.Label(second_pane, text="Second Pane", font=("Arial", 14))
second_pane_label.pack()
second_pane.pack_propagate(False)

# third pane
third_pane = tk.Frame(root, bg=style["pane"]["background-color"], borderwidth=pane_border_width, relief="solid", highlightbackground=pane_border_color, highlightthickness=pane_border_width)
third_pane.pack(fill=tk.BOTH, expand=True)
third_pane_label = tk.Label(third_pane, text="Third Pane", font=("Arial", 14))
third_pane_label.pack()
third_pane.pack_propagate(False)

# File dialog function
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("EDF Files", "*.edf")])
    if file_path:
        analysis_results_label.config(text=f"Uploaded EEG data: {file_path}")
# Handle the .edf file
        display_eeg_data(file_path)
    else:
        analysis_results_label.config(text="No EEG data uploaded.")

# Initialize a variable to hold the current EEG plot figure
first_pane_figure = None

# Start the tkinter main loop
root.mainloop()
