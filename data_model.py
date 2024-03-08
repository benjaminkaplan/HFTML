# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:25:51 2023

@author: benja
"""

def func(x):
    logging.info("in tje func")
    return x

import os
#os.chdir("C:/Binyamin/Thesis/HFT_ML")
from imports import *
class DataModel:
    
    def __init__(self, read_file=None, df=None, dfm=None, dfs=None, dft=None,
                 scaled_file=None, index=None, window=60, data_dir=None, 
                 desc=None, load=False, mask=None, train_pct=None, pickle_path = None, **kwargs):
        self.read_file = read_file
        self.scaled_file = scaled_file
        self.index = index
        self.window = window
        self.write_name = None
        self.data_dir = data_dir
        self.mask = mask
        self.train_pct = train_pct
        if self.data_dir is None:
            self.data_dir = os.path.join(".", "SavedData", f"data_{datetime.now().strftime('%Y_%m_%d')}")
        if pickle_path is not None:
            c = self.load(path=pickle_path, use_pickle = True)
            return
        
        
        if self.read_file is not None:
            if self.index is None or not isinstance(self.index, str):
                raise ValueError("Must Specify a single datetime index!")
            self.write_name = self.read_file.split(".")[0]
            df = pd.read_csv(read_file)
            df = df.set_index(index)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Index must be a datetime index!")
            self.df = df
        elif self.scaled_file is not None:
            self.write_name = self.scaled_file
            df = pd.read_csv(read_file)
            df.index = pd.MultiIndex.from_frame(df[index])
        elif df is not None:
            self.df = df
        elif dfs is not None:
            self.dfs=dfs
        elif dfm is not None:
            self.dfm = dfm
        elif self.data_dir is not None:
            self.load(data_dir, self.mask)
        if desc is not None:
            if self.write_name is None:
                self.write_name = desc
            self.desc = desc
            
    def create_multiindex(self,df=None, window=-1):
        logging.info("Creating MultiIndex...")
        if df is None:
            df = self.df.copy()
        if window == -1:
            window = self.window
        fake_window = window+1
        df['y_date'] = df.index.copy()
        df['x_dates'] = df.index.copy()
        l = [*df.rolling(window=fake_window)]
        print(f"total: {len(l)}")
        for i, df_temp in enumerate(l):
            if i %10000 == 0:
                print(i)
            df_temp['y_date'] = df_temp.iloc[-1].loc['y_date']
            df_temp.drop(df_temp.index[-1], inplace=True)
        df_temp = pd.concat(l[window:])
        df_temp = df_temp.set_index(['y_date', 'x_dates'])
        self.dfm = df_temp
        return self.dfm
    def make_y(self, df=None,shift=[-1], column = 'close', func=None):
        if df is None:
            df = self.dfm.copy()
        index_list = df.index.levels[0].to_list() if isinstance(df.index, pd.MultiIndex) else df.index.to_list()
        result_df = pd.DataFrame(index = index_list) 
        for s in shift:
            #y = df[column].groupby(level=0).tail(1).droplevel(level=1,axis=0) < df['close'].groupby(level=0).tail(1).shift(s).droplevel(level=1,axis=0)
            #y = y.replace({True:1, False:-1})
            y = df[column].groupby(level=0).tail(1).shift(s).droplevel(level=1,axis=0)
            result_df[s] = y
        result_df.index.name = "timestamp"
        self.dfy = result_df
        return self.dfy
    def scale_data(self, df=None, scaler=MinMaxScaler):
        logging.info("Begin Scaling Data...")
        idx = pd.IndexSlice
        if df is None:
            df = self.dfm.copy()
        scaled_df = df.copy()
        scaler_idx = df.index.get_level_values("y_date").to_list() if isinstance(df.index, pd.MultiIndex) else df.index.to_list()
        scaler_ser = pd.Series(index = scaler_idx, dtype=np.float64)
        cols = df.columns
        i = 0
        def scale(X):
            #print(X.index[0])
            X_ = np.atleast_2d(X)
            return pd.DataFrame(scaler().fit_transform(X_), columns = cols, index=X.index)
        scaled_df = df.groupby(level=0).apply(scale)
        self.dfs = scaled_df
        
        
        logging.info("Done Scaling Data.")
        return self.dfs
        
    def save(self, path=None, write_name=None, use_pickle=False):
        if path is None:
            path = self.data_dir
        if write_name is None:
            if self.write_name is None:
                write_name = input("Please provide a name for the data...")
            else:
                write_name = self.write_name
        Path(path).mkdir(parents=True, exist_ok=True)
        if pickle:
            logging.info(f"Saving pickle file {os.path.join(path, write_name)}...")
            with open(os.path.join(path, write_name+ ".pkl"), 'wb') as outp:
                pickle.dump(self.__dict__, outp, pickle.HIGHEST_PROTOCOL)
            logging.info("Done.")
            return self
        self.df.to_csv(os.path.join(path, write_name ))
        self.dfm.to_csv(os.path.join(path, write_name.split(".csv")[0] + "_multiindex.csv"))
        self.dfs.to_csv(os.path.join(path, write_name.split(".csv")[0] + "_multiindex_scaled.csv"))
        self.dfy.to_csv(os.path.join(path, write_name.split(".csv")[0] + "_y.csv"))
        self.dft.to_csv(os.path.join(path, write_name.split(".csv")[0] + "_multiindex_scaled_time.csv"))
    def load(self, path=None, mask=None, use_pickle=False):
        if use_pickle:
            logging.info(f"Loading data model stored in {path}...")
            with open(path, 'rb') as inp:
                temp_dict = pickle.load(inp)
                self.__dict__.update(temp_dict)
                logging.info("Loaded data model. Done.")
                return temp_dict
        if mask is not None:
            mask = os.path.join(path,mask.split(".csv")[0])
            df_name = mask + ".csv"
            dfy_name = mask+"_y.csv"
            dfm_name = mask+"_multiindex.csv"
            dfs_name = mask +"_multiindex_scaled.csv"
            dft_name = mask +"_multiindex_scaled_time.csv"
        else:
            files = os.listdir(path)
            files = [f for f in files  if ".csv" in f]
            y_file = [f for f in files if "_y.csv" in f]
            if len(y_file) == 1:
                dfy_name = y_file[0]
                files.remove(y_file[0])
            multiindex_file = [f for f in files if "_multiindex.csv" in f]
            if len(multiindex_file) == 1:
                dfm_name = pd.read_csv(multiindex_file[0])
                files.remove(multiindex_file[0])
            scaled_file = [f for f in files if "_multiindex_scaled.csv" in f]
            if len(scaled_file) == 1:
                dfs_name = pd.read_csv(scaled_file[0])
                files.remove(scaled_file[0])
            time_file = [f for f in files if "_multiindex_scaled_time.csv" in f]
            if len(time_file) == 1:
                    dft_name = pd.read_csv(time_file[0])
                    files.remove(time_file[0])
            if len(files) == 1:
                df_name =  pd.read_csv(files[0])
        df = pd.read_csv(df_name)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        logging.info(f"read in file: {df_name}")
        dfm = pd.read_csv(dfm_name)
        dfm['y_date'], dfm['x_dates'] = pd.to_datetime(dfm['y_date']), pd.to_datetime(dfm['x_dates'])
        dfm = dfm.set_index(pd.MultiIndex.from_frame(dfm[['y_date', 'x_dates']])).drop(['y_date', 'x_dates'], axis=1)
        logging.info(f"read in file: {dfm_name}")
        dfs = pd.read_csv(dfs_name)
        dfs['y_date'], dfs['x_dates'] = pd.to_datetime(dfs['y_date']), pd.to_datetime(dfs['x_dates'])
        dfs = dfs.set_index(pd.MultiIndex.from_frame(dfs[['y_date', 'x_dates']])).drop(['y_date', 'x_dates'], axis=1)
        logging.info(f"read in file: {dfs_name}")
        #dft = pd.read_csv(dft_name)
        #dft['y_date'], dfs['x_dates'] = pd.to_datetime(dft['y_date']), pd.to_datetime(dft['x_dates'])
        #dft = dfs.set_index(pd.MultiIndex.from_frame(dft[['y_date', 'x_dates']])).drop(['y_date', 'x_dates'], axis=1)
        #logging.info(f"read in file: {dfs_name}")
        dfy = pd.read_csv(dfy_name)
        dfy['timestamp'] = pd.to_datetime(dfy['timestamp'])
        dfy = dfy.set_index('timestamp')
        logging.info(f"read in file: {dfy_name}")
        self.df = df
        self.dfm = dfm
        self.dfy = dfy
        self.dfs = dfs
    
    
    def xytrain(self, pct=0.7, dft=None, dfy=None, shuffle = True, seed=1):
        idx = pd.IndexSlice
        if pct is None:
            pct = self.train_pct
        if dft is None:
            dft = self.dft.copy()
        if dfy is None:
            dfy = self.dfy.copy()
        y_dates_idx = dft.index.unique(level=0)
        y_dates_num = int(len(y_dates_idx) * pct)
        y_train_idx = y_dates_idx[:y_dates_num]
        y_test_idx = y_dates_idx[y_dates_num:]
        if shuffle:
            y_train_idx = y_train_idx.to_list()
            random.seed(seed)
            random.shuffle(y_train_idx)
        
        xtrain = dft.loc[idx[y_train_idx,:],:]
        ytrain = dfy.loc[y_train_idx]
        xtest = dft.loc[idx[y_test_idx],:]
        ytest = dfy.loc[y_test_idx]
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        return dict(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest)
    def truncate_and_stamp(self, df=None, level=1, type="linear"):
        if df is None:
            df = self.dfs.copy()
        t1 = datetime(2023,1,1,9,30)
        t2 = datetime(2023,1,1,16,0)
        num_mins = (t2-t1).seconds / 60.
        df = df.reset_index().set_index('y_date')
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        df = df.between_time("09:30","16:00")
        df = df.reset_index().set_index(['y_date','x_dates'])
        n_mins = ( pd.Timestamp("23:59")-pd.Timestamp("00:00")).seconds/60
        t_index = pd.date_range('00:00', '23:59', periods=n_mins+1, inclusive='left').time
        t_series = pd.Series(np.linspace(0,1,len(t_index)), name='time', index = t_index)
        df['x_time'] = pd.to_datetime(df.index.get_level_values(level), utc = True).tz_convert("America/New_york").time
        dft = df.reset_index().set_index('x_time', drop=False).join(t_series, how='left').set_index(['y_date', 'x_dates']).sort_values(['y_date', 'x_dates'])
        dft = dft.drop(['x_time'], axis=1)
        self.dft = dft
        return self.dft
    def do_all(self):
        print("Creating Multiindex ...")
        self.create_multiindex(window=60)
        print("Done Creadting Multiindex")
        print("Scaling Data...")
        self.scale_data()
        print("Done Scaling Data")
        print("Making Truncating, stamping and making y...")
        self.truncate_and_stamp()
        self.make_y(shift=[-1,-5,-10, -30, -60])
        self.xytrain()
        print("Done all.")
        
            
            
        
            
if __name__ == "__main__":
    #d = DataModel(read_file = "AAPL_minute_short.csv", index = "timestamp")
    from data_model import DataModel


    pickled = DataModel(read_file = "C:\Binyamin\Thesis\HFT_ML\AAPL_minute_short.csv", index = "timestamp")

    #d = DataModel(data_dir = "C:\Binyamin\Thesis\HFT_ML\SavedData\data_2023_02_09", mask = "AAPL_minute_short")
    #d.create_multiindex()
    #d.make_y()
    #d.scale_data()
    logging.info("hello world")