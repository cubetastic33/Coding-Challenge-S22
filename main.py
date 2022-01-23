from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from parse import parse_data

x, y = parse_data()

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

# Scale the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Create and train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate the model
score = model.score(x_test, y_test)
print(score)
