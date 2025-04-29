import psycopg2
import pandas as pd
from ssi_fc_data import fc_md_client, model
from config.config import *
from config import config_vn30_data as config
from datetime import datetime

class DataService:
    def __init__(self) -> None:
        # Instantiate the market data client for VN30 data.
        self.vn30_client = fc_md_client.MarketDataClient(config)
        print("DataService initialized")
            
    
    def get_vn30f_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch VN30f intraday OHLC data for the period between start_date and end_date.
        The start_date and end_date should be strings in "dd/mm/yyyy" format.
        
        The function splits the overall period by month (using fixed rules for month-end days)
        and then concatenates the data from each segment.
        """
        
        # Convert input strings from "yyyy-mm-dd" to "dd/mm/yyyy".
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d/%m/%Y")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y")
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        data_list = []

        # Iterate from the starting month/year to the ending month/year.
        current_year = start_dt.year
        current_month = start_dt.month

        while (current_year < end_dt.year) or (current_year == end_dt.year and current_month <= end_dt.month):
            # Determine the starting day for the current segment.
            day_start = start_dt.day if (current_year == start_dt.year and current_month == start_dt.month) else 1

            # Set the month's maximum day based on fixed rules.
            if current_month == 2:
                max_day = 28
            elif current_month in [4, 6, 9, 11]:
                max_day = 30
            else:
                max_day = 31

            # If this is the ending month, adjust the end day.
            day_end = end_dt.day if (current_year == end_dt.year and current_month == end_dt.month) else max_day

            # Format the date strings in "dd/mm/yyyy".
            month_str = str(current_month).zfill(2)
            seg_start_date = f"{str(day_start).zfill(2)}/{month_str}/{current_year}"
            seg_end_date = f"{str(day_end).zfill(2)}/{month_str}/{current_year}"
            print(seg_start_date, seg_end_date)
            # Fetch the VN30f data for this monthly segment.
            result = self.vn30_client.intraday_ohlc(
                config,
                model.intraday_ohlc('vn30f1m', seg_start_date, seg_end_date, 1, 9999, True, 1)
            )
            df = pd.DataFrame(result['data'])
            data_list.append(df)

            # Move to the next month.
            if current_month == 12:
                current_month = 1
                current_year += 1
            else:
                current_month += 1
                
        
        vn30f = pd.concat(data_list, ignore_index=True)
                
        #combine TradingDate and Time to datetime
        vn30f['datetime'] = pd.to_datetime(vn30f['TradingDate'] + ' ' + vn30f['Time'], format='%d/%m/%Y %H:%M:%S')
        vn30f = vn30f.set_index('datetime')
        vn30f = vn30f.drop(['TradingDate', 'Time', 'Value', 'Symbol'], axis=1)
        # Round the datetime to set the seconds to 0
        vn30f.index = vn30f.index.round('1T')

        # Change name of the remaining columns with no capitalization
        vn30f.columns = ['open', 'high', 'low', 'close', 'volume']
        

        # Concatenate all segments into a single DataFrame.
        return vn30f

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Merges the matched volume data with the VN30 data based on their datetime indexes.
        """
        matched_data = self.get_matched_data(start_date, end_date)
        vn30_data = self.get_vn30_data(start_date, end_date)
        # Merge the two dataframes by left join
        data = pd.merge(matched_data, vn30_data, how='left', left_index=True, right_index=True)
        # Rename the columns
        data.columns = ['open', 'high', 'low', 'close', 'volume', 'vn30']
        data.dropna(inplace=True)
        return data
    
    def get_train_data(self) -> pd.DataFrame:
        train = pd.read_csv("data/train.csv")
        # set datetime as index
        train.index = pd.to_datetime(train['datetime'])
        return train
    
    def get_test_data(self) -> pd.DataFrame:
        test = pd.read_csv("data/test.csv")
        # set datetime as index
        test.index = pd.to_datetime(test['datetime'])
        return test

# Instantiate the DataService.
data_service = DataService()

