import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from utils import create_features

def plot_predict_usage_model(df_all):
    _ = df_all[['AEP_MW','AEP_MW_Prediction']].plot(figsize=(15, 5))
    # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = df_all[['AEP_MW_Prediction','AEP_MW']].plot(ax=ax,
                                                style=['-','.'])
    ax.set_xbound(lower='01-01-2015', upper='02-01-2015')
    ax.set_ylim(0, 60000)
    plot = plt.suptitle('January 2015 Forecast vs Actuals')
    plt.show()

        # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = df_all[['AEP_MW_Prediction','AEP_MW']].plot(ax=ax,
                                                style=['-','.'])
    ax.set_xbound(lower='01-01-2015', upper='01-08-2015')
    ax.set_ylim(0, 60000)
    plot = plt.suptitle('First Week of January Forecast vs Actuals')
    plt.show()

def predict_usage_model():
    # Load the data
    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'AEP_Energy_consumption.xlsx'
    df = pd.read_excel(file_path, sheet_name="AEP_hourly",index_col=[0], parse_dates=[0])

    split_date = '01-Jan-2015'
    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()

    X_train, y_train = create_features(df_train, label='AEP_MW')
    X_test, y_test = create_features(df_test, label='AEP_MW')


    # Load the model
    model_file_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_xgb_model_usage.pkl'
    try:
        model = joblib.load(model_file_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_file_path}")
        return


    df_test['AEP_MW_Prediction'] = model.predict(X_test)
    df_all = pd.concat([df_test, df_train], sort=False)
  
    # Export to Excel
    # Export the Results to Excel
    base_dir_output = Path(__file__).resolve().parents[0]
    output_file_path = base_dir_output / 'output' / 'kubetune_recommended_usage.xlsx'
    df_all.to_excel(output_file_path,index=False)
    print(f"Predictions saved to {output_file_path}") 

    # print(f"Predictions saved to {output_file_path}")
    # Plot the results
    plot_predict_usage_model(df_all)
     
if __name__ == "__main__":
    predict_usage_model()
   