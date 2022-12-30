from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D
from keras.models import Model
import pandas as pd
import numpy as np
from get_feature_select import get_rrfeature,get_1rrfeature

neual_index=[1,2,3,4,5,6,7,8,9,10,11,12,13]
positive_index = [1,2,3,4,5,6,7,8,9,10,11,12,13]
negetive_index = [1,2,3,4,5,6,7,8,9,10,11,12,13]
test_neual_index=[10]
test_positive_index = [10]
test_negetive_index = [10]

neual_feature,neual_signal = get_rrfeature(index = neual_index , emotion_type= 'neual' , encode = 0)
positive_feature,positive_signal = get_rrfeature(index = positive_index , emotion_type= 'positive' , encode = 1)
negetive_feature,negetive_signal = get_rrfeature(index = negetive_index , emotion_type= 'negetive' , encode = 2)

all_feature = pd.concat([neual_feature , positive_feature,negetive_feature],ignore_index=1)

all_feature=all_feature.dropna(axis = 1)#清除nan值

test_neual_feature,test_neual_signal = get_1rrfeature(index = test_neual_index , emotion_type= 'neual' , encode = 0)
test_positive_feature,test_positive_signal = get_1rrfeature(index = test_positive_index , emotion_type= 'positive' , encode = 1)
test_negetive_feature,test_negetive_signal = get_1rrfeature(index = test_negetive_index , emotion_type= 'negetive' , encode = 2)

test_feature = pd.concat([test_neual_feature , test_positive_feature,test_negetive_feature],ignore_index=1)

test_feature=test_feature.dropna(axis = 1)#清除nan值

input_layer = Input(shape=(400, 16))
conv1 = Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu')(input_layer)
pool1 = MaxPooling1D(pool_size=4)(conv1)
lstm1 = LSTM(32)(pool1)
output_layer = Dense(400, activation='sigmoid')(lstm1)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=5)
# 模型評估，打分數
model.evaluate(x_test, y_test)