import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

print 'f'
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data = data[np.isnan(data['Age']) == False]
ldata = data[['Pclass', 'Fare', 'Age', 'Sex']].replace('female', 0).replace('male', 1).values
tdata = data['Survived'].values
# sex = data[(data['Sex'] == 'female')]['Name'].values
# s2 = []
# for s in sex:
#     pref = s.find('(')
#     fname = ""
#     if pref >= 0:
#         fname = s[pref+1:s.find(' ', pref)]
#     else:
#         dot = s.find('.')
#         lch = s.find(' ', dot+2)
#         fname = s[dot+2:lch]
#     s2.append(fname)
#
# Counter = Counter(s2)
#
# most_occur = Counter.most_common(1)
# print most_occur
print ldata
clf = DecisionTreeClassifier(random_state=241)
clf.fit(ldata, tdata)

print clf.feature_importances_