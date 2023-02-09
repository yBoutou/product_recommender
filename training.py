from processing.data_manager import save_model,load_model,save_files,read_files
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from processing.validation import clean_data
from aggregate import save_aggregated
import pandas as pd
import mlflow

from processing.data_manager import get_config

config = get_config()

orders_cleaned = config['orders_cleaned']
instacart_orders = config['instacart_orders']
transformed_orders = config['transformed_orders']
fp_growth_data = config['fp_growth_data']
association_rules_data = config['association_rules_data']


# Parameters
# MIN_SP = 0.001 # Minimum fp growth support   ## Commented on 6/12/2022
MIN_SP = config['MIN_SP'] # Minimum fp growth support
MIN_TH = config['MIN_TH']  # Minimum association rules threshold


orders = read_files(instacart_orders)

orders['product_id'] = orders['product_id'].astype('int')

#orders_clusters = pd.merge(orders,products_clusters[['product_id','cluster']],on='product_id',how='left')


df_orders_no_repetition = clean_data(orders)
#Saving cleaned data
save_files(df_orders_no_repetition,orders_cleaned)

def run_training():
    #mlflow.set_tracking_uri('http://localhost:5000')
    #mlflow.set_experiment(experiment_name='Product Based Association Rules')
    tags = {
        "dataset": "df_orders_binary",
        "release.version": "0.1"}

    with mlflow.start_run(run_name='fp_growth'):
        mlflow.set_tags(tags)

        save_aggregated(orders_cleaned)

        df_orders_binary = read_files(transformed_orders)

        df_orders_binary.dropna(axis=1,how='any',inplace = True)
        
        dataset = df_orders_binary[df_orders_binary.columns[1:]]
        
        dataset = dataset.astype('int')
        #dataset.drop(['product_id'],axis = 'columns',inplace=True)
        
        fpgrowth_res=fpgrowth(dataset,min_support=MIN_SP, use_colnames=True)

        save_model(fpgrowth_res,fp_growth_data)

        mlflow.log_param("min_support",MIN_SP)
        mlflow.log_artifact(fp_growth_data)

        df_orders_clusters_fpgrowth_pickle = load_model(fp_growth_data)

        print('df_orders_clusters_fpgrowth_pickle === ', df_orders_clusters_fpgrowth_pickle)

        df_association_rules=association_rules(df_orders_clusters_fpgrowth_pickle, metric="lift", min_threshold=MIN_TH)

        df_association_rules.sort_values(by=['conviction'],ascending=False,inplace = True, ignore_index=True)

        df_association_rules['antecedents'] = df_association_rules['antecedents'].apply(set)
        df_association_rules['consequents'] = df_association_rules['consequents'].apply(list)

        save_files(df_association_rules,association_rules_data)

        mlflow.end_run()

if __name__ == '__main__':
    run_training()