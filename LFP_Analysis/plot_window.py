import PanGUI
import DataProcessingTools as DPT
from DataProcessingTools.objects import DPObject
from pylab import gcf, gca
import numpy as np
import os
import pickle as pkl


class PlotObject(DPObject):
    argsList = ["data", ("title", "test")]

    def __init__(self, *args, **kwargs):
        DPObject.__init__(self, *args, **kwargs)
        self.data = self.args["data"]
        self.title = self.args["title"]
        self.indexer = self.getindex("trial")
        self.setidx = [0 for i in range(self.data.shape[0])]
        self.current_idx = None

    def plot(self, i=None, getNumEvents=False, getLevels=False, getPlotOpts=False, ax=None, **kwargs):
        """
        This function showcases the structure of a plot function that works with PanGUI.
        """
        # Define the plot options that this function understands
        plotopts = {
            "show": True, 
            "factor": 1.0, 
            "level": "trial", 
            "overlay": False,
            "second_axis": False, 
            "seeds": {"seed1": 1.0, "seed2": 2.0},
            "color": DPT.objects.ExclusiveOptions(["red", "green"], 0),
            "log_axis": False  # New option for logarithmic axis
        }
        if getPlotOpts:
            return plotopts

        # Extract the recognized plot options from kwargs
        for (k, v) in plotopts.items():
            plotopts[k] = kwargs.get(k, v)

        if getNumEvents:
            # Return the number of events available
            if plotopts["level"] == "trial":
                return self.data.shape[0], 0
            elif plotopts["level"] == "cell":
                if i is not None:
                    nidx = self.setidx[i]
                else:
                    nidx = 0
                return np.max(self.setidx)+1, nidx
            elif plotopts["level"] == "all":
                return 1, 0
        if getLevels:        
            # Return the possible levels for this object
            return ["trial", "cell", "all"]
        
        if plotopts["level"] == "all":
            idx = range(self.data.shape[0])
        elif plotopts["level"] == "cell":
            idx = np.asarray(self.setidx) == i
        else:
            idx = i
        if ax is None:
            ax = gca()
        if not plotopts["overlay"]:
            ax.clear()

        if plotopts["show"]:
            data = self.data[idx, :].T
            if len(data.shape) == 2:
                data = data.mean(1)
            f = plotopts["factor"]
            pcolor = plotopts["color"].selected()

            # Plot data
            ax.plot(f * data, color=pcolor)

            # Set logarithmic y-axis if requested
            if plotopts["log_axis"]:
                ax.set_yscale("log")

            # Add seeds as vertical lines
            ax.axvline(plotopts["seeds"]["seed1"])
            ax.axvline(plotopts["seeds"]["seed2"])

            if plotopts["second_axis"]:
                ax2 = ax.twinx()
                ax2.plot(0.5 * self.data[i, :].T, color="black")
        return ax

    def append(self, obj):
        DPObject.append(self, obj)
        self.data = np.concatenate((self.data, obj.data), axis=0)


def plot_processing(segment_df):
    """
    Create a PanGUI window for a 2D array extracted from `segment_df['segment']` and `segment_df['psd']`.
    """
    # Convert the segment and psd data to stacked 2D arrays
    pp1 = PlotObject(np.stack(segment_df['segment'].values), normpath=False, title="Segments")
    temp = np.array([spec[-1][:15] for spec in segment_df['flat_specs']])
    pp2 = PlotObject(temp, normpath=False, title="PSD")

    # Launch the PanGUI window with different options for each object
    ppg = PanGUI.create_window(
        [pp1, pp2],
        plotopts=[
            {"log_axis": False},  # No log axis for segments
            {"log_axis": True}    # Log axis for PSD
        ]
    )
    return ppg


file = '/Users/liuyuanwei/Documents/GitHub/Pyhipp_T/LFP_Analysis/Data Processed/20181105.pkl'

with open(file, 'rb') as f:
    segments_df = pkl.load(f)

plot_processing(segments_df)
