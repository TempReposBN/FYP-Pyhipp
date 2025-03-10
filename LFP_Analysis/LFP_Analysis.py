from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, 
                             QSpacerItem, QSizePolicy, QTextEdit, QFrame, QFileDialog)
import sys
import pandas as pd
from load_data import get_data
from processing_helpers import get_segments, get_power_spec, get_peak_fits
from tally_helper import TallyWindow
from plot_window import plot_processing
import pickle as pkl
import numpy as np
import DataProcessingTools as DPT


class MainWindow(QWidget):
    lfp_df = []
    ch_num_list = []
    lfp_mne = []
    session_start_time = 0
    markers = []
    timeStamps = []
    sampling_frequency = 0
    segments_df = pd.DataFrame()


    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set window properties
        self.setWindowTitle("LFP Analysis")
        self.setGeometry(100, 100, 600, 400)

        main_layout = QVBoxLayout()

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #C0C0C0;")

        #======Top layout for loading data ======
        top_layout = QHBoxLayout()

        date_label = QLabel("Date:")
        top_layout.addWidget(date_label)

        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("Enter date here")
        top_layout.addWidget(self.date_input)

        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_data)
        top_layout.addWidget(load_button)

        plot_mne_button = QPushButton("Plot Raw")
        plot_mne_button.clicked.connect(self.plot_mne)
        top_layout.addWidget(plot_mne_button)
        #========================================

        #====== Middle Layout for plotting ======
        processing_layout = QVBoxLayout()
        processing_layout.addWidget(line)
        processing_label = QLabel("Data Analysis")
        processing_layout.addWidget(processing_label)

        processing_child_layout = QHBoxLayout()
        window_size_label = QLabel("Window Size (ms):")
        processing_child_layout.addWidget(window_size_label)

        self.window_size_input = QLineEdit()
        self.window_size_input.setPlaceholderText("Enter Window Size (ms)")
        processing_child_layout.addWidget(self.window_size_input)

        get_segments_button = QPushButton("Get Segments")
        get_segments_button.clicked.connect(self.get_segment)
        processing_child_layout.addWidget(get_segments_button)

        tally_button = QPushButton("Show Tally")
        tally_button.clicked.connect(self.show_tally)
        processing_child_layout.addWidget(tally_button)

        save_pkl_button = QPushButton("Export to Pickle")
        save_pkl_button.clicked.connect(self.save_pkl)

        load_pkl_button = QPushButton("Load Pickle file")
        load_pkl_button.clicked.connect(self.load_pkl)
        
        plot_pangui_button = QPushButton("Plot in PanGUI")
        plot_pangui_button.clicked.connect(self.open_plot_window)


        processing_layout.addLayout(processing_child_layout)
        processing_layout.addWidget(save_pkl_button)
        processing_layout.addWidget(load_pkl_button)
        processing_layout.addWidget(plot_pangui_button)

        #========================================

        #======= Main Layout to nest all =======
        main_layout.addLayout(top_layout)
        main_layout.addLayout(processing_layout)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addItem(spacer)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        main_layout.addWidget(self.console_output)

        quit_button = QPushButton("Close")
        quit_button.clicked.connect(self.close)
        main_layout.addWidget(quit_button)
        self.setLayout(main_layout)
        #=======================================

    def load_data(self):
        # Get the date from the input field
        day = self.date_input.text()

        # Call the get_data function and display the output
        if day:
            self.console_output.append(f"Loading data for {day}...")
            self.lfp_df, self.ch_num_list, self.lfp_mne, self.session_start_time, self.markers, self.timeStamps, self.sampling_frequency= get_data(day)

            if self.lfp_df is not None:
                self.console_output.append(f"Loaded {len(self.ch_num_list)} channels.")
                self.console_output.append(str(self.lfp_df.head()))  # Display first few rows of the DataFrame
            else:
                self.console_output.append("Failed to load data.")
        else:
            self.console_output.append("Please enter a valid date.")
    

    def load_pkl(self):
        # Open a file dialog to select the Pickle file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Pickle File", "", "Pickle Files (*.pkl);;All Files (*)", options=options)

        if file_path:
            try:
                # Load the DataFrame from the Pickle file
                with open(file_path, 'rb') as file:
                    self.segments_df = pkl.load(file)

                # Optional: Update any UI components or console outputs as needed
                self.console_output.append(f"Data successfully loaded from {file_path}")
                self.console_output.append(f"Loaded data has {self.segments_df.shape[0]} rows and {self.segments_df.shape[1]} columns.")
                self.console_output.append(str(self.segments_df.head()))
            except Exception as e:
                self.console_output.append(f"Failed to load data: {str(e)}")
    

    def save_pkl(self):
        if self.segments_df.empty:
            self.console_output.append("No data available to export.")
            return

        # Open a file dialog to specify the Pickle file name and path
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Pickle File", "", "Pickle Files (*.pkl);;All Files (*)", options=options)

        if file_path:
            print(self.segments_df.dtypes)
            try:
                # Ensure the file has the .pkl extension
                if not file_path.endswith('.pkl'):
                    file_path += '.pkl'

                # Export the DataFrame to a Pickle file
                with open(file_path, 'wb') as file:
                    pkl.dump(self.segments_df, file)

                self.console_output.append(f"Data successfully exported to {file_path}")
            except Exception as e:
                self.console_output.append(f"Failed to export data: {str(e)}")

        
    def plot_mne(self):
        if self.lfp_mne:
            self.lfp_mne.plot()
        else: 
            self.console_output.append("LFP MNE object not found. Please load the data.")

    def open_plot_window(self):
        if self.segments_df.empty:
            self.console_output.append("No segments available to plot. Please generate segments first.")
            return

        self.plot_window = plot_processing(self.segments_df)
        self.plot_window.show()

    def get_segment(self):
        # Retrieve window size input
        try:
            window_size = int(self.window_size_input.text())
        except ValueError:
            self.console_output.append("Please enter a valid integer for window size.")
            return
        
        # Verify that data has been loaded
        if self.lfp_df.empty or not self.markers or not self.timeStamps:
            self.console_output.append("Please load data before extracting segments.")
            return

        self.segments_df = get_segments(self.lfp_df, self.markers, self.timeStamps, window_size=window_size)

        # Display output in console_output
        self.console_output.append(f"Extracted {len(self.segments_df)} segments.")

        frequencies = []
        psd_list = []
        p_stds_l = []
        p_means_l = []
        peak_gauss_l = []
        flat_spec_l = []

        count = 0

        # Convert df to array (faster)
        segments_list = self.segments_df['segment'].values

        for segment in segments_list:
            freq, psd = get_power_spec(segment, self.sampling_frequency, method='fft')
            frequencies.append(freq)
            psd_list.append(psd)

            p_stds, p_means, peak_gauss, flat_specs = get_peak_fits(psd, freq, [1,15])
            p_stds_l.append(p_stds)
            p_means_l.append(p_means)
            peak_gauss_l.append(peak_gauss)
            flat_spec_l.append(flat_specs)

            count+=1
            print(count)
        

        self.segments_df['psd'] = psd_list
        self.segments_df['freq'] = frequencies
        self.segments_df['peak_stds'] = p_stds_l
        self.segments_df['peak_means'] = p_means_l
        self.segments_df['peak_gauss'] = peak_gauss_l
        self.segments_df['flat_specs'] = flat_spec_l


        # self.console_output.append(f"Calculated PSD for {len(psd_list)} segments.")
        # self.console_output.append(str(self.segments_df.head()))  # Display first few rows


    def show_tally(self):
        if self.segments_df.empty:
            self.console_output.append("Please generate segments before viewing the tally.")
            return
        self.tally_window = TallyWindow(self.segments_df, self.ch_num_list)
        self.tally_window.show()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
