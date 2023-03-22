import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Wczytanie danych
data = pd.read_csv('lottery.csv')

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(data.drop('wynik', axis=1), data['wynik'], test_size=0.2)

# Utworzenie modelu
model = DecisionTreeClassifier()

# Trenowanie modelu
model.fit(X_train, y_train)

# Przewidywanie wyników dla zbioru testowego
y_pred = model.predict(X_test)

print(y_pred)
