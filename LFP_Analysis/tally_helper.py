from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QTextEdit, QHBoxLayout
import pandas as pd

class TallyWindow(QDialog):
    def __init__(self, segments_df, channels):
        super().__init__()
        self.segments_df = segments_df
        self.channels = channels
        self.filtered_df = segments_df

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Filter Segments")
        self.setGeometry(150, 150, 600, 400)

        layout = QVBoxLayout()
        checkbox_layout = QHBoxLayout()

        # Channel checkboxes
        channel_layout = QVBoxLayout()
        channel_layout.addWidget(QLabel("Select Channels:"))
        self.channel_checkboxes = []
        for channel in self.channels:
            checkbox = QCheckBox(f"Channel {channel}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.apply_filters)
            channel_layout.addWidget(checkbox)
            self.channel_checkboxes.append(checkbox)
        checkbox_layout.addLayout(channel_layout)

        # End Cue checkboxes
        end_cue_layout = QVBoxLayout()
        end_cue_layout.addWidget(QLabel("Select End Cues:"))
        self.end_cues = [31, 32, 33, 34, 35, 36]
        self.end_cue_checkboxes = []
        for cue in self.end_cues:
            checkbox = QCheckBox(f"End Cue {cue}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.apply_filters)
            end_cue_layout.addWidget(checkbox)
            self.end_cue_checkboxes.append(checkbox)
        checkbox_layout.addLayout(end_cue_layout)

        # Cue Onset checkboxes
        cue_onset_layout = QVBoxLayout()
        cue_onset_layout.addWidget(QLabel("Select Cue Onsets:"))
        self.cue_onsets = [11, 12, 13, 14, 15, 16]
        self.cue_onset_checkboxes = []
        for cue in self.cue_onsets:
            checkbox = QCheckBox(f"Cue Onset {cue}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.apply_filters)
            cue_onset_layout.addWidget(checkbox)
            self.cue_onset_checkboxes.append(checkbox)
        checkbox_layout.addLayout(cue_onset_layout)

        layout.addLayout(checkbox_layout)

        # Display filtered results
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(self.result_display)

        self.setLayout(layout)

        # Apply initial filters
        self.apply_filters()

    def apply_filters(self):

        # Get selected channels
        selected_channels = [
            checkbox.text().split(" ")[1][-3:] for checkbox in self.channel_checkboxes if checkbox.isChecked()
        ]

        # Get selected end cues
        selected_end_cues = [
            int(checkbox.text().split(" ")[2]) for checkbox in self.end_cue_checkboxes if checkbox.isChecked()
        ]
        print(f"Selected End Cues: {selected_end_cues}")

        # Get selected cue onsets
        selected_cue_onsets = [
            int(checkbox.text().split(" ")[2]) for checkbox in self.cue_onset_checkboxes if checkbox.isChecked()
        ]
        print(f"Selected Cue Onsets: {selected_cue_onsets}")

        # Convert channel column to string for comparison, if needed
        if 'channel' in self.segments_df.columns:
            self.segments_df['channel'] = self.segments_df['channel'].astype(str)

        # Ensure 'start_position' and 'cue_onset' columns are integers
        if 'start_position' in self.segments_df.columns:
            self.segments_df['start_position'] = self.segments_df['start_position'].astype(int)
        if 'cue_onset' in self.segments_df.columns:
            self.segments_df['cue_onset'] = self.segments_df['cue_onset'].astype(int)

        # Apply filters to the DataFrame with proper type handling
        self.filtered_df = self.segments_df[
            (self.segments_df['channel'].isin(selected_channels)) &
            (self.segments_df['start_position'].isin(selected_end_cues)) &
            (self.segments_df['cue_onset'].isin(selected_cue_onsets))
        ]

        # Debug output to check if filtering is working
        print(f"Filtered DataFrame:\n{self.filtered_df.head()}")

        # Update the display with the filtered results
        self.update_display()



    def update_display(self):
        """ Update the QTextEdit display with the filtered segments """
        # self.result_display.clear()
        total_segments = len(self.filtered_df)
        self.result_display.append(f"Total segments: {total_segments}")
