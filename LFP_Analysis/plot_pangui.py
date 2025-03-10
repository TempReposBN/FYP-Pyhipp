import PanGUI
import DataProcessingTools as DPT
from DataProcessingTools.objects import DPObject
from pylab import gcf, gca
import numpy as np
import scipy
import scipy.io as mio
import os

class PlotObject(DPObject):
    argsList = ["data", ("title", "test")]

    def __init__(self, df, column, *args, **kwargs):
        """
        Initialize PlotObject for plotting a specific column in the DataFrame.
        :param df: DataFrame containing the data.
        :param column: Column name to plot.
        """
        DPObject.__init__(self, *args, **kwargs)
        self.data = df[column].tolist()  # Store column data as a list
        self.column = column
        self.title = column  # Use the column name as the title
        self.indexer = self.getindex("index")
        self.setidx = [0 for i in range(len(self.data))]
        self.current_idx = None

    def plot(self, i=None, getNumEvents=False, getLevels=False, getPlotOpts=False, ax=None, **kwargs):
        """
        Plot the data for the specific column.
        """
        plotopts = {
            "show": True,
            "factor": 1.0,
            "level": "index",
            "overlay": False,
            "color": DPT.objects.ExclusiveOptions(["blue", "red", "green"], 0),
        }
        if getPlotOpts:
            return plotopts

        for k, v in plotopts.items():
            plotopts[k] = kwargs.get(k, v)

        if getNumEvents:
            return len(self.data), 0

        if getLevels:
            return ["index"]

        if i is None or i >= len(self.data):
            i = 0  # Default to the first index if out of range

        if ax is None:
            ax = gca()

        if not plotopts["overlay"]:
            ax.clear()

        if plotopts["show"]:
            if self.column == "segment":
                ax.plot(self.data[i], color=plotopts["color"].selected(), label=self.column)
            elif self.column == "psd":
                ax.plot(self.data[i], color=plotopts["color"].selected(), label=self.column)
            elif self.column == "flat_specs":
                ax.plot(self.data[i], color=plotopts["color"].selected(), label=self.column)
            elif self.column == "peak_gauss":
                for peak in self.data[i]:  # Assuming `peak_gauss` is a list of lists
                    ax.plot(peak, linestyle="dashed", label=f"{self.column} Peak")
            ax.legend()
            ax.set_title(f"{self.title} (Index {i})")
        return ax

def plot_segments_df(segments_df):
    plotobjs = [
        PlotObject(segments_df, "segment"),
        PlotObject(segments_df, "psd"),
        PlotObject(segments_df, "flat_specs"),
        PlotObject(segments_df, "peak_gauss")
    ]
    PanGUI.create_window(plotobjs)