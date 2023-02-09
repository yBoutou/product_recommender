import pandas as pd
from processing.data_manager import save_files
from processing.features import split_to_batchs,fake_order,one_hot_encoding,fix_columns
from processing.data_manager import get_config

config = get_config()
NUM_BATCHES = config['NUM_BATCHES']
transformed_orders = config['transformed_orders']

def  save_aggregated(data):

    count = 1
    df_batches = split_to_batchs(data,NUM_BATCHES)
    fake_order_df = fake_order(data)

    for batch in df_batches:    
        # For testing
        # batch = batch.head(1000)  

        # Remove the unreqired index column from each batch 
        batch.drop(['index'], axis=1, inplace=True)
        
        ##################################### Appending the fake order at the start of each batch ###############
        batch = batch.append(fake_order_df)

        ################################### One Hot Encoding ###################################

        encoder,encoder_df = one_hot_encoding(batch)

        df_orders_binary = batch.join(encoder_df)

        print('Batch '+str(count) +' Done\n')
        
        #print('df_orders_binary.columns == ',df_orders_binary.columns)
        

        # Drop unrequired product column 
        df_orders_binary.drop(['product_id'], axis=1, inplace=True)
    

        ############################ Change Column Names of the Output Dataframe #################################
        feature_names_list = encoder.get_feature_names_out()
        
        #print('feature_names_list == ',feature_names_list)
        df_orders_binary.columns = fix_columns(feature_names_list)

        
        ################################# Aggregation ###################################

        print('Aggregation Started...')
        
        print("df_orders_binary==== ",df_orders_binary.loc[:, ~df_orders_binary.columns.duplicated()])
        df_orders_binary = df_orders_binary.loc[:, ~df_orders_binary.columns.duplicated()]
        df_orders_no_repetition_ordered_agg = df_orders_binary.groupby('order_id').agg('max').reset_index()
        df_orders_no_repetition_ordered_agg = df_orders_no_repetition_ordered_agg.dropna(how="any")
        print('Aggregation Ended\n')
        
        ################################# Concatenation of aggregated dataframes ###################################    
        df_batches[count-1] = df_orders_no_repetition_ordered_agg
        count += 1
        #display(df_orders_no_repetition_ordered_agg.head(5))

    # Concatenate the batches in one dataframe
    df_orders_binary_concatenated = pd.concat(df_batches)

    # Removing the fake order rows before saving
    df_orders_binary_concatenated = df_orders_binary_concatenated[df_orders_binary_concatenated.order_id != -1]

    # Saving the transformed data
    save_files(data=df_orders_binary_concatenated,save_path=transformed_orders)