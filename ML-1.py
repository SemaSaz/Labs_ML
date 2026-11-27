import pandas as pd

data = pd.read_csv('titanic_train.csv', index_col='PassengerId')

print("=== Задание 1 ===")
print(data['Sex'].value_counts().to_string())

print()
print("=== Задание 2 ===")
print(data['Pclass'].value_counts().sort_index().to_string())
print(pd.crosstab(data['Pclass'], data['Sex']))

print()
print("=== Задание 3 ===")
print(round(data['Fare'].median(), 2))
print(round(data['Fare'].std(), 2))

print()
print("=== Задание 4 ===")
print(data.groupby('Survived')['Age'].mean().round(2).to_string())

print()
print("=== Задание 5 ===")
print(round(data[data['Age'] < 30]['Survived'].mean() * 100, 1))
print(round(data[data['Age'] > 60]['Survived'].mean() * 100, 1))

print()
print("=== Задание 6 ===")
print((data.groupby('Sex')['Survived'].mean() * 100).round(1).to_string())

print()
print("=== Задание 7 ===")
names = data[data['Sex'] == 'male']['Name'].str.split(',').str[1]
names = names.str.split('.').str[1].str.strip()
print(names.str.split(' ').str[0].value_counts().head().to_string())

print()
print("=== Задание 8 ===")
print(data.groupby(['Pclass', 'Sex'])['Age'].mean().unstack().round(1))
print(data.groupby('Pclass')['Age'].mean().round(1).to_string())