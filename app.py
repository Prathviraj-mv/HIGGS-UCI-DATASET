from Helpers.data_extract_and_preprocessing import DATA
from Helpers.plotters import  PLOT

class APP_:
    def __init__(self):

        data = DATA()
        data.run_ep_print()
        data.run_ep_function()

        plot =PLOT()
        plot.plot_correlation()


