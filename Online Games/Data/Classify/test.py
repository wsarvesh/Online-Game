# binary classification, breast cancer dataset, label and one hot encoded
import numpy
import pickle
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# load data
data = read_csv('breast-cancer.csv', header=None)
dataset = data.values
X = dataset[:,0:9]
print()
X = X.astype(str)
print()
Y = dataset[:,9]

# encode string input values as integers
encoded_x = None
print(X[len(X) - 1], len(X))
print(type(X), X.shape)
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
	feature = onehot_encoder.fit_transform(feature)
	if encoded_x is None:
		encoded_x = feature
	else:
		encoded_x = numpy.concatenate((encoded_x, feature), axis=1)

# 40-49,premeno,15-19,0-2,yes,3,right,left_up,no
# 40-49,premeno,30-34,0-2,no,1,left,left_low,yes

print("X shape: : ", encoded_x)
# encode string class values as integers
label_encoder_pred = LabelEncoder()
label_encoder_pred = label_encoder_pred.fit(Y)
print(label_encoder)
label_encoded_y = label_encoder_pred.transform(Y)
numpy.save('classes.npy', label_encoder_pred.classes_)
print(label_encoded_y)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

pred = input("Enter value: ").split(",")
X = numpy.append(X, [pred], axis=0)
print(X[len(X) - 1], len(X))
pred_x= None
# print(len(le), len(oe), range(0, preda.shape[1]))
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
	feature = onehot_encoder.fit_transform(feature)
	if pred_x is None:
		pred_x = feature
	else:
		pred_x = numpy.concatenate((pred_x, feature), axis=1)

inp = numpy.array([pred_x[len(pred_x) - 1]])
print(type(inp), inp.shape)


encoder = LabelEncoder()
encoder.classes_ = numpy.load('classes.npy', allow_pickle=True)

o_pred = model.predict(inp)
# print(o_pred, type(enc))
print(encoder.inverse_transform(o_pred))
