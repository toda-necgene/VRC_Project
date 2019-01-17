import matplotlib.pyplot as plot

class WavePlot():
    def __init__(self):
        pass

    def savefig(self, file, plotters):
        plot.clf()
        plot.subplot(2, 1, 1)
        plot.imshow(plotters[0], vmin=-15, vmax=5, aspect="auto")
        plot.subplot(2, 1, 2)
        plot.plot(plotters[1])
        plot.ylim(-1, 1)
        plot.savefig(file)
        
