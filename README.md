# Machine-Learning-Hotel-Cancellation-Prediction

A significant number of hotel bookings are called off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with. Such losses are particularly high on last-minute cancellations.

The new technologies involving online booking channels have dramatically changed customers’ booking possibilities and behavior. This adds a further dimension to the challenge of how hotels handle cancellations, which are no longer limited to traditional booking and guest characteristics.

The cancellation of bookings impacts a hotel on various fronts:
1. Loss of resources (revenue) when the hotel cannot resell the room.
2. Additional costs of distribution channels by increasing commissions or paying for publicity to help sell these rooms.
3. Lowering prices last minute, so the hotel can resell a room, reducing the profit margin.
4. Human resources to make arrangements for the guests.

The increasing number of cancellations calls for a Machine Learning based solution that can help predict which booking is likely to be canceled. INN Hotels Group has a chain of hotels in Portugal, they are facing problems with the high number of booking cancellations and have reached out to your firm for data-driven solutions. You as a data scientist have to analyze the data provided to find which factors have a high influence on booking cancellations, build a predictive model that can predict which booking is going to be canceled in advance, and help in formulating profitable policies for cancellations and refunds.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
http://localhost:8888/notebooks/Untitled70.ipynb?kernel_name=python3#
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,plot_confusion_matrix,precision_recall_curve,roc_curve,make_scorer

import warnings;
import numpy as np
warnings.filterwarnings('ignore')
hotel = pd.read_csv('/Users/marcusdoty/Downloads/INNHotelsGroup.csv')
data = hotel.copy()
data.head()

data.tail()

data.shape

data.info()

data.duplicated().sum()

data = data.drop(['Booking_ID'], axis=1)
data.head()

data.describe()

#An average of just under two adults seem to be cancelling and 
#about 1/10 kids per cancellation. They seem to be staying less
#than one weekend day on average and just over two days on average.
#Typically, it would be rare for them to require parking. 
#interestingly, they would order quite some time in advance before
#the booking. For the most part, they would not be repeated guests
#and would spend around $100 on a room per night.

def hist_box(data,col):
  f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12,6))
 
  sns.boxplot(data[col], ax=ax_box, showmeans=True)
  sns.distplot(data[col], ax=ax_hist)
  plt.show()
hist_box(data, 'lead_time')

#The mode of this data is very close to 0, indicating that the
#time of paying was quite close to their stay, the data is also
#heavy on the right-tail, indicating that there are less larger
#lead times. This tells us, for the most part most cancellations
#are not large overall, with a median of around 3 months. This is 
#evident with around 50% of cancellations occuring around 6 weeks
#and less. 

hist_box(data, 'avg_price_per_room')

#The average price of the data tends to be more dense in regards
#to distribution with a high peak and bell-shaped curve around the
#$100 mark. Half of the cancellations occur at between the $80 to 
#120 mark. Much of the cancellations also occur in very cheap hotel
#rooms with many close to $0, incidating complimentary rooms.

data[data['avg_price_per_room'] == 0]

data.loc[data['avg_price_per_room'] == 0, 'market_segment_type'].value_counts()

data.loc[data['avg_price_per_room'] >= 500, 'avg_price_per_room'] = Upper_Whisker
sns.countplot(data['no_of_children'])
plt.show()

data['no_of_children'].value_counts(normalize=True)

data['no_of_children'] = data['no_of_children'].replace([9, 10], 3)
sns.countplot(data['arrival_month'])
plt.show()

data['arrival_month'].value_counts(normalize=True)

sns.countplot(data['booking_status'])
plt.show()

data['booking_status'].value_counts(normalize=True)

data['booking_status'] = data['booking_status'].apply(
    lambda x: 1 if x == 'Canceled' else 0
)
cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(data.corr(), annot=True, cbar=True, cmap='jet')
plt.show()

#The strongest evident correlations exist between previous guest
#and number of previously not-cancelled hotel rooms, indicating 
#that with previous guests tend to not be cancelling hotels rooms.
#Those who have children seem to be spending more on hotel rooms.
#Furthermore, there are not an overbearing amount of strong 
#correlations.

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data, x='market_segment_type', y='avg_price_per_room', palette='gist_rainbow'
)

def stacked_barplot(data,predictor,target,figsize=(10,6)):
  (pd.crosstab(data[predictor],data[target],normalize='index')*100).plot(kind='bar',figsize=figsize,stacked=True)
  plt.legend(loc="lower right")
  plt.ylabel('Percentage Cancellations %')
stacked_barplot(data, 'market_segment_type', 'booking_status')

#Of the percentage cancellation rate, the largest tend to occur
#online, with aviation and offline being slightly behind. We
#can also see corporate cancellations seem to have a 10%
#cancellation rate with none occuring complimentary. 

stacked_barplot(data, 'repeated_guest', 'booking_status')

#Repeated guests tend to do cancellations at a tiny rate, less 
#than 2%, whilst over 30% of newly-visiting guests are cancelling.

stay_data = data[(data["no_of_week_nights"] > 0) & (data["no_of_weekend_nights"] > 0)]
stay_data["total_days"] = (stay_data["no_of_week_nights"] + stay_data["no_of_weekend_nights"])

stacked_barplot(stay_data, "total_days", "booking_status",figsize=(15,6))

plt.figure(figsize=(10, 5))
sns.lineplot(y=data["avg_price_per_room"], x=data["arrival_month"], ci=None)
plt.show()

X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,stratify=Y, random_state=1)
print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))

def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Cancelled', 'Cancelled'], yticklabels=['Not Cancelled', 'Cancelled'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
lg = LogisticRegression()
lg.fit(X_train,y_train)

y_pred_train = lg.predict(X_train)
metrics_score(y_train, y_pred_train)

y_pred_test = lg.predict(X_test)
metrics_score(y_test, y_pred_test)

y_scores_lg=lg.predict_proba(X_train)

precisions_lg, recalls_lg, thresholds_lg = precision_recall_curve(y_train, y_scores_lg[:,1])

plt.figure(figsize=(10,7))
plt.plot(thresholds_lg, precisions_lg[:-1], 'b--', label='precision')
plt.plot(thresholds_lg, recalls_lg[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()

optimal_threshold = 0.3

y_pred_train = lg.predict_proba(X_train)
metrics_score(y_train, y_pred_train[:,1]>optimal_threshold)

y_pred_test = lg.predict_proba(X_test)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold)

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train_scaled = scaling.transform(X_train)
X_test_scaled = scaling.transform(X_test)

svm = SVC(kernel='linear',probability=True)
model = svm.fit(X= X_train_scaled, y = y_train)

y_pred_train_svm = model.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)

y_pred_test_svm = model.predict(X_test_scaled)
metrics_score(y_test, y_pred_test_svm)

y_scores_svm=model.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm[:,1])

plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label='recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()

optimal_threshold_svm= 0.4

y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train_svm[:,1]>optimal_threshold_svm)

y_pred_test = model.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold_svm)

svm_rbf=SVC(kernel='rbf',probability=True)
svm_rbf.fit(X_train_scaled,y_train)

y_pred_train_svm = svm_rbf.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)

y_pred_test = svm_rbf.predict(X_test_scaled)
metrics_score(y_test, y_pred_test)

y_scores_svm=svm_rbf.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm[:,1])

plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label='recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()

optimal_threshold_svm=0.3

y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train_svm[:,1]>optimal_threshold_svm)

y_pred_test = svm_rbf.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold_svm)

model_dt = DecisionTreeClassifier(random_state=1)
model_dt.fit(X_train, y_train)

pred_train_dt = model_dt.predict(X_train)
metrics_score(y_train, pred_train_dt)

pred_test_dt = model_dt.predict(X_test)
metrics_score(y_test, pred_test_dt)

estimator = DecisionTreeClassifier(random_state=1)

parameters = {
    'max_depth': np.arange(2, 7, 2),
    'max_leaf_nodes': [50, 75, 150, 250],
    'min_samples_split': [10, 30, 50, 70],
}

grid_obj = GridSearchCV(estimator, parameters, cv=5,scoring='recall',n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

estimator = grid_obj.best_estimator_

estimator.fit(X_train, y_train)

dt_tuned = estimator.predict(X_train)
metrics_score(y_train,dt_tuned)

y_pred_tuned = estimator.predict(X_test)
metrics_score(y_test,y_pred_tuned)

feature_names = list(X_train.columns)
plt.figure(figsize=(20, 10))
out = tree.plot_tree(
    estimator,max_depth=3,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)

for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(1)
plt.show()

importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

rf_estimator = RandomForestClassifier( random_state = 1)

rf_estimator.fit(X_train, y_train)

y_pred_train_rf = rf_estimator.predict(X_train)

metrics_score(y_train, y_pred_train_rf)

y_pred_test_rf = rf_estimator.predict(X_test)

metrics_score(y_test, y_pred_test_rf)

importances = rf_estimator.feature_importances_

columns = X_train.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize = (13, 13))

sns.barplot(importance_df.Importance, importance_df.index)
