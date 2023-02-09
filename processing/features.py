import numpy as np
import pandas as pd
from processing.data_manager import read_files
from sklearn.preprocessing import OneHotEncoder

from processing.data_manager import get_config

config = get_config()



def split_to_batchs(data,NUM_BATCHES):
    #Splitting the data into batches
    df_orders_no_repetition_ordered = read_files(data)

    unique_orders_id = df_orders_no_repetition_ordered['order_id'].unique()
    #unique_products_id = df_orders_no_repetition_ordered['product_id'].unique()
    unique_order_ids_split = np.array_split(unique_orders_id, NUM_BATCHES)


    start = 0
    df_batches = []
    """
    to split the data accordingly we have to get the index of the first and last unique order-id and get the corresponding data
    to fill the next batch, the start index will automatically be the (end-index of the last batch + 1)
    """
    for unique_id_list in unique_order_ids_split:
        # The end index of each batch is determined by the index of the last unique id in the corresponding unique id list 
        end_index = df_orders_no_repetition_ordered.index[df_orders_no_repetition_ordered['order_id']==unique_id_list[-1]][-1]
        
        # Append the extracted batches into a huge list 
        df_batches.append(df_orders_no_repetition_ordered.iloc[start:end_index+1].reset_index())
        
        # Update the start index of the next batch
        start = end_index + 1
    
    return df_batches


def fake_order(data): 
    df_orders_no_repetition_ordered = read_files(data)
    ############# numpy array for fake order with ID -1 where it contains all products ##############

    fake_order= [[-1, x] for x in df_orders_no_repetition_ordered]
    fake_order = np.array(fake_order)
    fake_order_df = pd.DataFrame(fake_order)
    fake_order_df.columns = ['order_id', 'product_id']
    return fake_order_df


def one_hot_encoding(batch):
    encoder = OneHotEncoder(handle_unknown='error')
    print('batch.dtypes==== ',batch.dtypes)
    print('batch.dtypes==== ',batch.dtypes)
    #batch.apply(pd.to_numeric)
    #batch.astype(np.float16)
    batch[['product_id']] = batch[['product_id']].astype(str)
    batch[['order_id']] = batch[['order_id']].astype(str)
    print('batch.dtypes==== ',batch.dtypes)
    encoder_df = pd.DataFrame(encoder.fit_transform(batch[['product_id']]).toarray())
    
    return encoder, encoder_df


def fix_columns(feature_names):
    columns_names = ['order_id']
    
    # To change the name of column from product_X to 39 only for example, so we remove the first 7 letters and the last 2 letters
    product_names =  [feature[11::] for feature in feature_names ]

    columns_names.extend(product_names)
    
    return columns_names