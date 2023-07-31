import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from statsmodels.nonparametric.kernel_regression import KernelReg

nums = [str(i) for i in range(1,13)]
letts = list('ABCDEFGH')

class PlateAssay:

    """
    A Python data structure for timecourse data collected on a 96-well plate.
    Intended to make it easy to get different views and selections of the data.
    """

    def __init__(self, dfs, df_keys=[],
                row_annotations = {}, col_annotations = {}, annotations = {},
                time_method="average"):

        self._datasets = {}
        self.keys = list(df_keys)
        self._times = []
        self._raw_data = dfs
        
        self.row_names = np.unique([i[0] for i in dfs[0].columns[2:]])
        self.col_names = np.unique([i[1:] for i in dfs[0].columns[2:]])
        
        self.n_rows = len(self.row_names); self.n_cols = len(self.col_names)

        if isinstance(dfs, (list, tuple)):

            if len(df_keys)==0:
                df_keys = np.arange(len(dfs))

            for n, df in enumerate(dfs):

                DataArr = np.zeros((len(df), self.n_rows, self.n_cols))

                for nt in range(len(df)):
                    for nl,lett in enumerate(self.row_names):
                        for nn, num in enumerate(self.col_names):

                            try:
                                DataArr[(nt, nl, nn)] = float(df.iloc[nt][lett + str(num)])
                            except:
                                DataArr[(nt, nl, nn)] = float(str(df.iloc[nt][lett + str(num)]).replace('*',''))

                self._datasets[df_keys[n]] = DataArr
                self._times.append(np.unique(df["Time"]))
            if time_method == 'average':
                self.times = np.mean(np.array(self._times), axis=0)
            else:
                self.times = np.array(self._times)

        else:
            key=df_keys; df = dfs; self.keys = [key]
            DataArr = np.zeros((len(df), self.n_rows, self.n_cols))

            for nt in range(len(df)):
                for nl,lett in enumerate(self.row_names):
                    for nn, num in enumerate(self.col_names):

                        try:
                            DataArr[(nt, nl, nn)] = float(df.iloc[nt][lett + str(num)])
                        except:
                            DataArr[(nt, nl, nn)] = float(str(df.iloc[nt][lett + str(num)]).replace('*',''))
            self.times = np.unique(df["Time"])
            self._datasets[key] = DataArr

        self.annotations = annotations
        self._row_annotations = row_annotations
        self._col_annotations = col_annotations

        for key, construct in row_annotations.items():
            self.annotations[key] = np.array([construct]*self.n_cols).T

        for key, construct in col_annotations.items():
            self.annotations[key] = np.array([construct]*self.n_rows)

        self.annotation_types = list(self.annotations.keys())

        self.make_df()

    def make_df(self):

        df_holder = []

        for t in range(len(self.times)):
            for nr,row in enumerate(self._datasets[self.keys[0]][t]):
                for nc,value in enumerate(row):
                    x = [self.times[t], nr, nc]
                    an_names = []

                    for key in self.annotation_types:
                        try:
                            x.append(self.annotations[key][(nr, nc)])
                        except IndexError:
                            raise IndexError("Annotation '"+key+"' is of wrong length")

                    for key in self.keys:
                        x.append(self._datasets[key][(t,nr,nc)])

                    df_holder.append(x)

        col_names = ["time", "row", "col"] + list(self.annotation_types) + list(self.keys)
        self.df = pd.DataFrame(df_holder, columns = col_names)


    def __getitem__(self, index):

        if isinstance(index, str):

            return self._datasets[index]

        elif isinstance(index, slice):

            return PlateAssay([dataset.iloc[index] for dataset in self._raw_data],
                              df_keys = self.keys, annotations=self.annotations)

#     def __iter__(self):

#         return __iter__(self._datasets)

    def __setitem__(self, key, values):

        self._datasets[key] = values
        if key not in self.keys:
            self.keys.append(key)
        self.make_df()

    def __len__(self):

        return self.__getitem__(self.keys[0]).shape[0]

    def select(self, index_dict):
        return selector(self.df, index_dict)

    def grid(self, y, hue=None, col=None, row=None, x="time", aspect=1.5, height=2.5,
                palette=None, log_hue=False, hue_zero_val=0):

        """Plot a seaborn FacetGrid showing the timecourses across samples. Allows for
            easy averaging over replicates and separation based on a number of different variables. Takes:

            MANDATORY:
            y: the coordinate of the graphs, should be one of the experimental datasets you put in

            OPTIONAL:
                *Which data to plot*
                x: the x coordinates of the graphs (DEFAULT: "time")
                hue: how to separate out curves within a single graph
                col: how to split out columns, if desired (default None will make only one column)
                row: how to split out rows, if desired (default None will make only one row)

                *How to color it*
                palette: which palette to use. defaults to None, whatever your seaborn default is
                    note: I like to use a range from a 'light' to 'dark' value of a color, like:
                    palette = sns.blend_palette(['lightgreen', 'darkgreen'])
                    (otherwise, the lowest value is 0, which you can't see on the graph!)
                log_hue: whether to take the logarithm of the hue value (DEFAULT: False)
                hue_zero_val: if there are zeros in the value being used for the hue and you're taking
                    the log, you'll need to set a new value because -Inf is for losers (DEFAULT: 0)

                * How big to make the plot*
                aspect: the aspect ratio of each plot, defaults to 1.5
                height: the height of each graph, defaults to 2.5

        """

        df = self.df

        # Automatically select only rows and columns of relevance
        # (I use the use_row and use_col annotations here for that)
        if "use_col" in self.annotations and "use_row" in self.annotations:
            df = selector(df, {"use_col":True, "use_row":True})

        # I often want to plot my hues on a log scale, so I added this to do that properly
        if log_hue:

            # In the new copied dataframe, reset the hue to be the log of the hue
            df['hue'] = np.log(df[hue])

            # If the hue was zero (now -inf), set it to hue_zero_val (default 0)
            df['hue'][df['hue']==-np.inf] = hue_zero_val

        else:
            # otherwise, df[hue] is just whatever column you said to be the hue
            df['hue'] = df[hue]

        # define a facetgrid in seaborn
        g = sns.FacetGrid(df, col=col, row=row, margin_titles=True, aspect=aspect, height=height)

        # map a lineplot onto that
        g.map(sns.lineplot, x, y, 'hue', palette=palette)

        return g

    def draw(self, row_annotations=None, col_annotations=None, colormap="Greys", linecolor='grey', lw=2, sep=' / '):

        '''Draw a cartoon of the plate setup based on your annotations as a reference.'''

        PlateImage = np.zeros((8, 12))
        if "use_col" in self.annotations:
            PlateImage[self.annotations['use_col']==False] = 1
            use_col = self._col_annotations['use_col']
        else:
            use_col = np.arange(12)

        if "use_row" in self.annotations:
            PlateImage[self.annotations['use_row']==False] = 1
            use_row = self._row_annotations['use_row']
        else:
            use_row = np.arange(8)

        plt.imshow(PlateImage, cmap=colormap, vmax=1)
        for i in range(12):
            plt.axvline(i+0.5, c=linecolor, lw=lw)
        for i in range(8):
            plt.axhline(i+0.5, c=linecolor, lw=lw)

        if isinstance(col_annotations, str):
            plt.xticks(np.arange(12)[use_col], np.array(self._col_annotations[col_annotations])[use_col],
                       rotation=90)
            plt.xlabel(col_annotations)

        elif isinstance(col_annotations, (list, tuple, np.ndarray)):
            colkeys = []
            for i in np.where(use_col)[0]:
                colkeys.append(sep.join([str(self._col_annotations[ann][i]) for ann in col_annotations]))
            plt.xticks(np.arange(12)[use_col], colkeys, rotation=90)
        else:
            plt.xticks([])

        if isinstance(row_annotations, str):
            plt.yticks(np.arange(8)[use_row], np.array(self._row_annotations[row_annotations])[use_row])
            plt.ylabel(row_annotations)

        elif isinstance(row_annotations, (list, tuple, np.ndarray)):
            rowkeys = []
            for i in np.where(use_row)[0]:
                rowkeys.append(sep.join([self._row_annotations[ann] for ann in row_annotations]))
            plt.yticks(np.arange(12)[use_row], rowkeys)
        else:
            plt.yticks([])
def selector(df, dic):

    '''A little tool for selecting from pandas dataframes by passing a dictionary, e.g.
            selector(df, {"color":"red", "shape":["square", "circle"]})

        For advanced usage, you can pass a function and it will return where True, e.g.
            selector(df, ["name": lambda name: "Sam" in name])

        You can also use this to select things greater than or less than a value, e.g.
            selector(df, ["enrichment": lambda enr: enr > 1])'''

    X = df.copy()

    for key,val in dic.items():

        # If you pass a tuple, list or numpy array
        if isinstance(val, (tuple, list, np.ndarray)):
            where = np.any(np.array([X[key]==v for v in val]),axis=0)
            X = X.loc[where]

        # If you pass a function
        elif isinstance(val, type(lambda x: x+1)):
            X = X.loc[X[key].apply(val)]

        # Otherwise we assume it's a single value
        else:
            X = X.loc[X[key]==val]

    return X


def read_Mn_assay_excel(filename, sheets = ("Plate 1", "Plate 2", "Plate 3"), directory="./", t_offset = 5, initial_guess=20, n_guesses=30):

    '''A data structure to read in data from my Mn2+ import assay'''

    data_holder = []

    for n,sheet in enumerate(sheets):

        # Read in the excel page for that sheet
        X = load_neo_data(directory + filename, sheet, initial_guess=initial_guess, n_guesses=n_guesses)
        
        # If the data holder is empty (this is the first sheet we're adding from)
        #   we don't need to set a time offset
        if len(data_holder)==0:
            t_off = 0

        # otherwise we'll add the previous times onto our time measurement
        #  in addition to an 'offset' for the time not counted between sheets
        #  (defaults to 5 minutes)
        else:
            t_off = np.max(data_holder[-1]["Time"]) + n*t_offset

        # Conver the time measurement into minutes
        X["Time"] = [time.hour*60 + time.minute + time.second / 60 + t_off for time in X["Time"] if type(time)!=str]
        data_holder.append(X)

    # concatenate all data into a single data frame
    X = pd.concat(data_holder)

    # create a unified index for the datapoints
    X.index = np.arange(len(X))

    return X


def load_neo_data(filename, sheet_name, initial_guess=20, n_guesses=30):

    '''If you're really lazy you can use this function to automatically figure out the header length and number of data cols to include.
         Especially useful if you're trying to load in many sheets at once and don't want to manually get the values for each of them.'''

    N = 0

    # Try every possible header length from initial_guess to initial_guess + n_guesses
    for n in range(initial_guess, initial_guess + n_guesses):

        # Try reading in as a pandas dataframe with the correct sheet
        test_df = pd.read_excel(filename, sheet_name=sheet_name, header=n)
        
        
        if "A1" in test_df.columns:
            N = n

    if N==0:
        raise Exception("can't find the right column!!")
    else:

        # load in actual neo data, correct sheet now that we know the header
        neo_df = pd.read_excel(filename, sheet_name=sheet_name, header=N)

        # loop through columns
        col_list = []
        for col in neo_df.columns:

            # if it's a datetime, add it
            if type(neo_df[col].iloc[0])==datetime.time:
                col_list.append(col)

            else:
                # if the first entryp isn't nan, add it
                if not np.isnan(neo_df[col].iloc[0]):
                    col_list.append(col)

        # take the relevant columns we identified above
        neo_df = neo_df[col_list]

        # return and drop nans from the end
        return neo_df.dropna(how = 'any')

def OD_RFU_Kernel_Regression(baseline_ODs, baseline_RFUs, kernel_bw=0.1, plot_expand=1.1):
    
    '''
    Performs a nonparametric kernel regression on the ODs against the RFU values and returns
    a function to normalize all sample RFUs against RFUs for a similar OD value in the EV/EV sample.
    
    I am doing it this way because ODs and RFUs in the control samples (no FP) do not obey a linear or
    any other kind of simple functional form. This way I do not assume any functional form of the input;
    I simple flit a flexible gaussian kernel function to the points.
    
    Inputs:
        baseline ODs: the control ODs (e.g. the EV/EV sample), as a 1D array
        baseline RFUs: the corresponding RFUs from this sample
        
        kernel_bw (optional): the bandwidth of the kernel to use. this controls the smoothness of the function.
            statsmodels has default procedures to estimate this but I like to manually set a significantly higher
            one because I feel that their procedures overfit based on systematic noise (in particular, they assume
            that the noise is random, while the noise is in fact highly correlated since it is coming from time course
            data). Defaults to 0.1. Feel free to play with this.
            
    Output:
        kr_predict(): a function that takes in any array of OD values and spits back out the kernel regression
            prediction of what the RFUs would be for the baseline (EV/EV?) sample. Built to handle >1D arrays
            as well as nans (nans in the input will just remain nans in the output)
        a plot of the raw normalization data (gray scatter) and the kernel regression fit (red line)
    
    WARNING:
        this does NOT work if any of your sample ODs are outside of the range of the EV/EV ODs. If you wish to use
        this function in case, you have to truncate your data such that the sample ODs are all less than the EV/EV
        ODs. As you can probably see from the plot, the regressor will drift back toward ZERO above the range of the
        fitting data. It CANNOT extrapolate to larger values.
    '''

    # define and fit the kernel gression
    kr = KernelReg(exog=baseline_ODs,
                   endog=baseline_RFUs,
                   var_type="c",
                   bw=[kernel_bw])

    pred_y, marginal_effects = kr.fit()
    
    # make the data to plot
    plot_x = np.arange(0, np.max(baseline_ODs)*plot_expand, 0.01)
    kr_prediction_means = kr.fit(plot_x)[0]
    
    # plot the fit
    plt.figure(figsize=(5,5))
    plt.plot(plot_x, kr_prediction_means, c='red')
    plt.scatter(baseline_ODs, baseline_RFUs, c='gray', alpha=0.1)
    plt.xlabel("Baseline OD"); plt.ylabel("Baseline RFU")
    
    # defines a function to return
    def kr_predict(ods):
        
        '''
        This is the output function of the regression protocol and is what the function returns. Use this to
        normalize your raw data.
        
        Input:
            ods: a numpy array of OD values of any shape
        Output:
            rfu_pred: predicted RFU values, as an array of the same shape. nans will be carried over
        '''
        
        flattened_ods = ods.flatten()
        
        rfu_pred = np.zeros(len(flattened_ods))
        nonan = np.isnan(flattened_ods)==False
        rfu_pred[nonan] = kr.fit(flattened_ods[nonan])[0]
        rfu_pred[nonan==False] = np.nan
        
        return rfu_pred.reshape(ods.shape)
    
    return kr_predict
