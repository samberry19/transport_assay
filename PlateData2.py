import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import datetime

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

        if isinstance(dfs, (list, tuple)):

            if len(df_keys)==0:
                df_keys = np.arange(len(dfs))

            for n, df in enumerate(dfs):

                DataArr = np.zeros((len(df), 8, 12))

                for nt in range(len(df)):
                    for nl,lett in enumerate(list('ABCDEFGH')):
                        for nn, num in enumerate(range(1,13)):

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
            DataArr = np.zeros((len(df), 8, 12))

            for nt in range(len(df)):
                for nl,lett in enumerate(list('ABCDEFGH')):
                    for nn, num in enumerate(range(1,13)):

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
            self.annotations[key] = np.array([construct]*12).T

        for key, construct in col_annotations.items():
            self.annotations[key] = np.array([construct]*8)

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

        if "use_col" in self.annotations and "use_row" in self.annotations:
            df = selector(df, {"use_col":True, "use_row":True})

        if log_hue:
            df['hue'] = np.log(df[hue])
            df['hue'][df['hue']==-np.inf] = hue_zero_val
        else:
            df['hue'] = df[hue]

        g = sns.FacetGrid(df, col=col, row=row, margin_titles=True, aspect=aspect, height=height)
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
                rowkeys.append(sep.join([self._col_annotations[ann] for ann in row_annotations]))
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
        X["Time"] = [time.hour*60 + time.minute + time.second / 60 + t_off for time in X["Time"]]
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

        # Try reading in as a pandas dataframe
        test_df = pd.read_excel(filename, sheet_name=sheet_name, header=n)
        
        
        if "A1" in test_df.columns:
            N = n

    if N==0:
        raise Exception("can't find the right column!!")
    else:

        datx = pd.read_excel(filename, header=N)

        col_list = []
        for col in datx.columns:

            if type(datx[col].iloc[0])==datetime.time:
                col_list.append(col)

            else:
                if not np.isnan(datx[col].iloc[0]):
                    col_list.append(col)

        daty = datx[col_list]

        datz = daty.loc[np.array([type(i)==int for i in daty[daty.columns[-1]]])]

        return datz
