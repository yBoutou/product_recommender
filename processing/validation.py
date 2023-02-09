import pandas as pd

def clean_data(orders):

    #Removing duplicates
    df_droplog = pd.DataFrame()  # Not required, used only for knowing the number of rows removed
    mask = orders.duplicated(subset=['order_id','product_id'], keep='first')
    df_orders_unified = orders.loc[~mask]

    df_droplog = df_droplog.append(orders.loc[mask])


    #Drop unrequired columns
    df_orders_no_repetition = df_orders_unified.drop(columns=['add_to_cart_order','reordered'])    #Drop Nan
    
    df_orders_no_repetition = df_orders_no_repetition.dropna(how="any")

    return df_orders_no_repetition