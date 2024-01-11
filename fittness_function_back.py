import os
import numpy as np
import pandas as pd
import keras

def calculate_fitness(data):
    print('GNN')

if __name__ == "__main__":
    
    # Load dataImport numpy as np
    np.ramdom.seed(42)
    # https://www.kaggle.com/code/groznykon/predicting-house-prices-with-a-neural-network
    script_dir = os.path.dirname(__file__)
    train = pd.read_csv(os.path.join(script_dir,'data/train.csv'))
    test = pd.read_csv(os.path.join(script_dir,'data/test.csv'))
    
    # From https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python.
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    train_targets = train.SalePrice
    train_data = train[features]
    test_data = test[features]
    
    # Preprocess data.
    # mean = train_data.mean(axis=0)
    # std = train_data.std(axis=0)
    # train_data = (train_data - mean) / std
    # test_data = (test_data - mean) / std
    
    # # Handling NaN values
    # mean2 = test_data['GarageCars'].mean()
    # mean3 = test_data['TotalBsmtSF'].mean()

    # test_data = np.array(test_data)
    # test_data[1116][2] = mean2
    # test_data[660][3] = mean3
    
    # # Build the model.
    # from keras import models
    # from keras import layers
    # from keras.layers import BatchNormalization
    # from keras import optimizers

    # model = models.Sequential()
    # model.add(layers.Dense(32, activation='relu', input_shape=(train_data.shape[1],)))
    # model.add(BatchNormalization())
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(layers.Dense(1))

    # model.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])
    # model.save(os.path.join(script_dir,'model'))
    # # model = keras.modelsk.load_model('path/to/location')
    
    # # Train model.
    # history = model.fit(train_data, train_targets, validation_split=0.2, epochs=2000, batch_size=32, verbose=0)

    # Plot learning history.
    # import matplotlib.pyplot as plt
    # plt.plot(history.history['loss'], 'r')
    # plt.plot(history.history['val_loss'], 'b')
    # plt.plot(history.history['mean_absolute_error'], 'r')
    # plt.plot(history.history['val_mean_absolute_error'], 'b')
    model = keras.models.load_model(os.path.join(script_dir,'model'))
    # Get predictions.
    predicted_prices = model.predict(test_data)
    print(predicted_prices)
    
    predicted_prices = np.squeeze(predicted_prices)
    predicted_prices = np.array(predicted_prices, dtype='float64')
    

   