import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

def plot_scrollable(data_dict):
    # Create a tkinter window
    root = tk.Tk()
    root.title('Scrollable Plot')

    # Create a scrollable frame
    scrollable_frame = ttk.Frame(root)
    scrollable_frame.grid(row=0, column=0, sticky='nsew')

    # Create a canvas to hold the plot
    canvas = tk.Canvas(scrollable_frame)
    canvas.grid(row=0, column=0, sticky='nsew')

    # Create a figure for the plot
    fig = plt.Figure(figsize=(8, 4))

    # Add subplot for each channel
    subplots = []
    for i, (channel_name, channel_data) in enumerate(data_dict.items()):
        ax = fig.add_subplot(len(data_dict), 1, i+1)
        ax.plot(channel_data)
        ax.set_title(channel_name)
        subplots.append(ax)

    # Create a tkinter canvas widget for the plot
    canvas = FigureCanvasTkAgg(fig, master=canvas)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Create a scrollbar for the canvas
    scrollbar = ttk.Scrollbar(scrollable_frame, orient='vertical', command=canvas.get_tk_widget().yview)
    scrollbar.grid(row=0, column=1, sticky='ns')
    canvas.get_tk_widget()['yscrollcommand'] = scrollbar.set

    # Configure resizing behavior
    scrollable_frame.rowconfigure(0, weight=1)
    scrollable_frame.columnconfigure(0, weight=1)

    # Run the tkinter event loop
    root.mainloop()
