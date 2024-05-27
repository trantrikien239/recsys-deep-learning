"""
This module contains the Order class, which is used to represent an order-based
dataset. The implication is that train/val/test split are sliced by order of 
the same user.
"""
from ..utils import timeit, list_concat
import pandas as pd

from .base import BaseDataset
from torch.utils.data import Dataset as TorchDataset

class OrderDataset(BaseDataset):
    def __init__(self, 
                 src_path,
                 order_file="orders.csv",
                 order_item_file="order_products__prior.csv",
                 user_col="user_id",
                 order_col="order_id",
                 order_number_col="order_number",
                 item_col="product_id"
                 ) -> None:
        super().__init__(src_path)
        self.order_file = order_file
        self.order_item_file = order_item_file
        self.user_col = user_col
        self.order_col = order_col
        self.order_number_col = order_number_col
        self.item_col = item_col
        
    @timeit
    def __init_order(self, excl_tags_col=None, excl_tags_values=None):
        self.excl_tags_col = excl_tags_col
        self.excl_tags_values = excl_tags_values

        if self.order_file.endswith(".csv"):        
            self.order_data = pd.read_csv(f"{self.src_path}/{self.order_file}")
        elif self.order_file.endswith(".parquet"):
            self.order_data = pd.read_parquet(f"{self.src_path}/{self.order_file}")
        else:
            raise ValueError("Only csv and parquet files are supported, "
                             f"got {self.order_file} instead.")

        if self.excl_tags_values and self.excl_tags_col:
            self.order_data = self.order_data[
                ~self.order_data[self.excl_tags_col].isin(self.excl_tags_values)]

        self.order_data = self.order_data[
            [self.user_col, self.order_col, self.order_number_col]]
        self.order_data = self.order_data.sort_values(
            [self.user_col, self.order_number_col])
        
        self.order_data_grouped = self.order_data.groupby(self.user_col)[
            self.order_number_col].max().reset_index()
        self.order_data = self.order_data.merge(
            self.order_data_grouped, on=self.user_col, suffixes=("", "_max")
        )
        self.order_data['order_number_reverse'] = self.order_data[
            self.order_number_col + '_max'] - self.order_data[self.order_number_col
        ]
        self.order_data.drop([self.order_number_col, self.order_number_col + '_max'], 
                             axis=1, inplace=True)
        
    @timeit
    def __split_order(self, shift=0, n_test=1, n_val=1):
        self.shift = shift
        self.n_test = n_test
        self.n_val = n_val

        self.order_data["split"] = "train"
        self.order_data.loc[
            self.order_data["order_number_reverse"].between(
                shift, shift + n_test - 1
                ), "split"
        ] = "test"
        self.order_data.loc[
            self.order_data["order_number_reverse"].between(
                shift + n_test, shift + n_test + n_val - 1
                ), "split"
        ] = "val"

    @timeit
    def __load_order_item(
            self, 
            item_sort_by='add_to_cart_order', 
            features=None
            ) -> None:
        if self.order_item_file.endswith(".csv"):
            self.order_item_data = pd.read_csv(
                f"{self.src_path}/{self.order_item_file}"
            )
        elif self.order_item_file.endswith(".parquet"):
            self.order_item_data = pd.read_parquet(
                f"{self.src_path}/{self.order_item_file}"
            )
        else:
            raise ValueError("Only csv and parquet files are supported, "
                             f"got {self.order_item_file} instead.")
        
        self.item_id_range = (self.order_item_data[self.item_col].min(),
                                self.order_item_data[self.item_col].max())
        self.item_id_nunque = self.order_item_data[self.item_col].nunique()

        self.order_item_data.sort_values(
            [self.order_col, item_sort_by], inplace=True
            )
        
        self.oi_features =  [self.item_col]
        if features:
            self.oi_features = self.oi_features + features
        
        self.order_item_data = self.order_item_data.groupby('order_id')[
            self.oi_features].agg(list).reset_index()
        self.order_item_data['cnt_items'] = self.order_item_data[
            self.oi_features[0]].apply(len)
        

    def load(self) -> None:
        self.__init_order(
            excl_tags_col='eval_set', 
            excl_tags_values=['train', 'test'])
        print("Order data loaded. Schema:")
        print(self.order_data.dtypes)
        self.__load_order_item(
            item_sort_by='add_to_cart_order', 
            features=['reordered']
        )
        print("Order item data loaded. Schema:")
        print(self.order_item_data.dtypes)

    def transform(self, shift=0, n_test=1, n_val=1, split_list=None) -> None:
        if not split_list:
            split_list = ['train']
        
        self.__split_order(shift=shift, n_test=n_test, n_val=n_val)
        df_merged_ = self.order_data.merge(
            self.order_item_data, on=self.order_col
        )
        df_merged_ = df_merged_[df_merged_['split'].isin(split_list)]
        df_merged_ = df_merged_.sort_values(
            [self.user_col, 'order_number_reverse'], ascending=[True, False]
        )
        df_out_ = df_merged_[self.user_col].drop_duplicates().to_frame()\
            .reset_index(drop=True)
        for feat in self.oi_features:
            df_out_ = df_out_.merge(df_merged_.groupby(self.user_col)[feat].agg(
                lambda x: list_concat(x, sep=0)
                ).to_frame(), on=self.user_col)
        df_out_['cnt_items'] = df_out_[self.oi_features[0]].apply(len)
        return df_out_
    
class NextItemDataset(TorchDataset):
    def __init__(self, df_order_item_users, item_id_range, max_seq_len=1024):
        self.df_order_item_users = df_order_item_users.reset_index(drop=True)
        self.item_id_range = item_id_range
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.df_order_item_users)
    
    def __getitem__(self, idx):
        seq_ = self.df_order_item_users.loc[idx, 'product_id']
        if len(seq_) > self.max_seq_len:
            seq_ = seq_[:self.max_seq_len]
        return seq_