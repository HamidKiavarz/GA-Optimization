import os
import numpy as np
import pandas as pd
import keras

def calculate_Score_fitness(featute_list):
    print('GNN')
def save_to_csv(data,result):
    # Make data frame of above data
    df = pd.DataFrame([data])
    # append data frame to CSV file
    df.to_csv(result, mode='a', index=False, header=False)
 
if __name__ == "__main__":
    
    # Load data
    # https://www.kaggle.com/code/groznykon/predicting-house-prices-with-a-neural-network
    # Normalization : https://developers.google.com/machine-learning/data-prep/transform/normalization
    script_dir = os.path.dirname(__file__)
    train = pd.read_csv(os.path.join(script_dir,'data/trapelo_train.csv'))
    test = pd.read_csv(os.path.join(script_dir,'data/trapelo_test.csv'))
    result = os.path.join(script_dir,'data/result_8000.csv')
    # From https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python.
    features = ['Degree','TWA','EWA','IWA','EWINA','Floor_Area','Lighting_Load','Plug_Load','Roof_Area','Type','EWR','EWINR','IWR','People_Density','People','People_Vent','Area_Vent_Rate','Total_Fresh_Air']
    train_feature = train.Total_Energy_Normal
    expected = test.Total_Energy_Normal
    train_data = train[features]
    test_data = test[features]
    print(expected)
    # Preprocess data.
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    # # Handling NaN values
    # mean2 = test_data['GarageCars'].mean()
    # mean3 = test_data['TotalBsmtSF'].mean()

    test_data = np.array(test_data)
    # test_data[1116][2] = mean2
    # test_data[660][3] = mean3
    
    # # Build the model.
    from keras import models
    from keras import layers
    from keras.layers import BatchNormalization
    from keras import optimizers
    # example of calculate the mean squared error
    from sklearn.metrics import mean_squared_error
    # create prediction model ---------------
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(BatchNormalization())
    model.add(layers.Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])
    np.random.seed(42)
    epoch = 8000
    # for epoch in range(2000,3000,5):
    # # Train model.
    
    history = model.fit(train_data, train_feature,
                        validation_split=0.2, epochs=epoch,
                        batch_size=32, verbose=0)
    model.save(os.path.join(script_dir,'model_orginal'))
    weighs = model.get_weights()
    
    #Load fitted model -----------
    
    # model = keras.models.load_model(os.path.join(script_dir,'model'))
    
    # Plot learning history.
    # import matplotlib.pyplot as plt
    # plt.plot(history.history['loss'], 'r')
    # plt.plot(history.history['val_loss'], 'b')
    # plt.plot(history.history['mean_absolute_error'], 'r')
    # plt.plot(history.history['val_mean_absolute_error'], 'b')

    # Get predictions.
    predicted = model.predict(test_data)
    
    # calculate errors
    error = mean_squared_error(np.array(expected), predicted)
    print('%s - %s' %(epoch,error))
    # data of Player and their performance
    data = {'epoch': epoch,'error': error}
    array = np.column_stack([expected, predicted])
    df = pd.DataFrame(array)
    df.to_csv( index=False)
    save_to_csv(data,result)
    # predicted_energy = np.squeeze(predicted)
    # predicted_energy = np.array(predicted, dtype='float64')


   