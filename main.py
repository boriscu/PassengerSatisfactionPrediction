import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)

# Početno preprocesiranje podataka (proveriti da li postoje nedostajuće vrednosti ili anomalije, pretvoriti stringove u brojeve, izbaciti neke atribute ako je očigledno da ne utiču na formiranje izlaza i sl.)
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Izbor numeričkih kolona
numeric_columns = ['Age', 'Flight Distance',
                   'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# Pregled podataka
train_data.info()
test_data.info()

# Prve dve kolone su beskorisne (Unamed, id)
train_data = train_data.drop(train_data.iloc[:, [0, 1]], axis=1)
test_data = test_data.drop(test_data.iloc[:, [0, 1]], axis=1)

# Provera balansiranosti skupa - Vidimo da nam je skup poprilicno izbalansiran
plt.pie(train_data.satisfaction.value_counts(), labels=[
        "Neutral or dissatisfied", "Satisfied"], colors=sns.color_palette("Greens"), autopct='%1.1f%%')
plt.show()

# Provera nedostajucih vrednosti
print(train_data.isnull().sum())
print("#######################")
print(test_data.isnull().sum())

# Popunjavanje nedostajucih vrednosti koje se nalaze samo u koloni arrival delay in minutes
#  Medijana je otporna na ekstremne vrednosti
train_data['Arrival Delay in Minutes'].fillna(
    train_data['Arrival Delay in Minutes'].median(), inplace=True)
test_data['Arrival Delay in Minutes'].fillna(
    test_data['Arrival Delay in Minutes'].median(), inplace=True)

# Provera korelacija numerickih vrednosti
# Izračunavanje korelacione matrice
correlation_matrix = train_data[numeric_columns].corr()

# Prikazivanje korelacione matrice kao toplotne mape
sns.heatmap(correlation_matrix, annot=True, cmap='Greens')
plt.title('Korelaciona matrica numeričkih promenljivih')
plt.show()


# Scatter plot prikaz zavisnosti dve kolone
plt.scatter(train_data['Arrival Delay in Minutes'],
            train_data['Departure Delay in Minutes'], alpha=0.5, color='green')
plt.show()


# Nakon primecivanja jake korelacije izmedju arrival i departure delay dropujemo arrival delay
train_data = train_data.drop('Arrival Delay in Minutes', axis=1)
test_data = test_data.drop('Arrival Delay in Minutes', axis=1)

# Update numerickih kolona
numeric_columns = ['Age', 'Flight Distance',
                   'Departure Delay in Minutes']

# Vecina putnika koji su leteli economy ili economy plus klasom su bili nezadovoljni, a biznis su zadovoljni
sns.countplot(x='Class', hue='satisfaction', palette="Greens", data=train_data)
plt.show()


# Svi koji su rejtovali wifi sa  5 zvezdica su zadovoljni
sns.countplot(x='Inflight wifi service', hue='satisfaction',
              palette="Greens", data=train_data)
plt.show()


# Procenat zadovoljstva putnika po polu
gender_counts = train_data['Gender'].value_counts()
satisfaction_counts = train_data['satisfaction'].value_counts()

satisfied_gender_counts = train_data[train_data['satisfaction']
                                     == 'satisfied']['Gender'].value_counts()

satisfied_percentages = satisfied_gender_counts / gender_counts * 100

plt.bar(satisfied_percentages.index, satisfied_percentages.values,
        color=['green', 'darkgreen'])
plt.title('Procenat zadovoljstva putnika prema polu')
plt.xlabel('Pol')
plt.ylabel('Procenat')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.ylim(0, 100)
plt.show()


# Zadovoljstvo i godine
fig, axes = plt.subplots(3, 1, figsize=(8, 15))

# Zadovoljstvo putnika po godinama
age_satisfaction = train_data.groupby('Age')['satisfaction'].value_counts(
    normalize=True).unstack().reset_index()

axes[0].plot(age_satisfaction['Age'],
             age_satisfaction['satisfied'] * 100, label='Satisfied', color='green')
axes[0].set_title('Zadovoljstvo putnika u odnosu na starost')
axes[0].set_xlabel('Starost')
axes[0].set_ylabel('Procenat zadovoljstva')
axes[0].legend()

# Ocene "Leg room service" po godinama
age_legroom = train_data.groupby(
    'Age')['Leg room service'].mean().reset_index()

axes[1].plot(age_legroom['Age'],
             age_legroom['Leg room service'], color='green')
axes[1].set_title('Ocene "Leg room service" u odnosu na starost')
axes[1].set_xlabel('Starost')
axes[1].set_ylabel('Ocena "Leg room service"')

# Ocene "Inflight wifi service" po godinama
age_wifi = train_data.groupby(
    'Age')['Inflight wifi service'].mean().reset_index()

axes[2].plot(age_wifi['Age'], age_wifi['Inflight wifi service'], color='green')
axes[2].set_title('Ocene "Inflight wifi service" u odnosu na starost')
axes[2].set_xlabel('Starost')
axes[2].set_ylabel('Ocena "Inflight wifi service"')

plt.tight_layout()  # Za bolji raspored subplotova
plt.show()


# Raspodela klasa letenja po godinama
age_class = train_data.groupby(['Age', 'Class']).size().unstack()

age_class.plot(kind='bar', stacked=True, color=[
               "#92D050", "#00B050", "#00B0F0"])
plt.title('Raspodela klasa letenja u odnosu na starost')
plt.xlabel('Starost')
plt.ylabel('Broj putnika')
plt.legend(title='Klasa letenja')
plt.show()


# Broj lojalnih putnika po godinama
sns.histplot(train_data, x="Age", hue="Customer Type",
             multiple="stack", palette="Greens", edgecolor=".3", linewidth=.5)
plt.show()


# One-hot kodiranje za nezavisne kategoričke kolone
train_data = pd.get_dummies(
    train_data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])

# Label encoding za ciljnu promenljivu
le = LabelEncoder()
train_data['satisfaction'] = le.fit_transform(train_data['satisfaction'])

# One-hot kodiranje za nezavisne kategoričke kolone u testnom skupu
test_data = pd.get_dummies(
    test_data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])

# Label encoding za ciljnu promenljivu u testnom skupu
test_data['satisfaction'] = le.transform(test_data['satisfaction'])

# Brojanje anomalija
anomaly_counts = {}

# Primenjivanje Z-Score metode na svaku numeričku kolonu
for column in numeric_columns:
    z_scores = np.abs(stats.zscore(train_data[column]))
    # Označavamo vrednosti koje su više od 3 standardne devijacije daleko od srednje vrednosti kao anomalije
    anomalies = (z_scores > 3)
    anomaly_counts[column] = anomalies.sum()  # Brojimo anomalije

print("Anomaly count: ", anomaly_counts)

plt.figure(figsize=(8, 6))
sns.histplot(train_data['Departure Delay in Minutes'],
             bins=30, kde=True, color='green')
plt.xlabel('Departure Delay in Minutes')
plt.ylabel('Frekvencija')
plt.title('Distribucija Departure Delay in Minutes')
plt.show()


# Primeti se velik broj anomalija u departure delay pa nju transformisemo
train_data['Departure Delay in Minutes'] = np.log(
    train_data['Departure Delay in Minutes'] + 1)

test_data['Departure Delay in Minutes'] = np.log(
    test_data['Departure Delay in Minutes'] + 1)

# Ponovno racunanje anomalija za kriticnu kolonu
z_scores = np.abs(stats.zscore(train_data['Departure Delay in Minutes']))
anomalies = (z_scores > 3)
print(
    f"Number of anomalies in {'Departure Delay in Minutes'} after transformation: {anomalies.sum()}")

plt.figure(figsize=(8, 6))
sns.histplot(train_data['Departure Delay in Minutes'],
             bins=30, kde=True, color='green')
plt.xlabel('Departure Delay in Minutes')
plt.ylabel('Frekvencija')
plt.title('Distribucija Departure Delay in Minutes')
plt.show()

# Kreiranje modela koji će vršiti klasifikaciju (obratiti pažnju na sve faktore koji utiču na to da li će neki algoritam dati dobre rezultate za određeni problem)
models = [LogisticRegression(), KNeighborsClassifier(),
          RandomForestClassifier(), DecisionTreeClassifier()]

y_train = train_data["satisfaction"]
X_train = train_data.drop("satisfaction", axis=1)
y_test = test_data["satisfaction"]
X_test = test_data.drop("satisfaction", axis=1)

# Izbor modela - RandomForestClassifier i DecisionTree daju najbolje rezultate
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print("Za model ", type(model).__name__, " rezultati su: ")
    print("\tTacnost: ", accuracy_score(y_test, y_predict))
    print("\tPreciznost: ", precision_score(y_test, y_predict))
    print("\tOdziv: ", recall_score(y_test, y_predict))
    print("\tF1 score: ", f1_score(y_test, y_predict))

    cm = confusion_matrix(y_test, y_predict)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Greens', ax=axs[i // 2, i % 2])
    axs[i // 2, i % 2].set_title(type(model).__name__)
    axs[i // 2, i % 2].set_xlabel('Predicted')
    axs[i // 2, i % 2].set_ylabel('True')
    print("\tMatrica konfuzije: ", cm)

    print("\tUnakrsna validacija: ")
    # Izvršavanje kros-validacije sa 5 foldova
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    # Prikaz rezultata svakog folda
    for fold, score in enumerate(scores):
        print(f"\t\tRezultat za fold {fold+1}: {score}")
    # Prikaz prosečne tačnosti
    average_accuracy = scores.mean()
    print(f"\t\tProsečna tačnost: {average_accuracy}")

plt.tight_layout()
plt.show()

# RandomForest vs DecisionTree, podesavanje hiperparametara

param_grid_rf = {
    'n_estimators': [100, 500, 1000, 1500, 2000],
    'max_depth': [10, 40, 80, 100, None],
    'min_samples_leaf': [1, 2],
}

param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0, 0.001, 0.01, 0.1, 0.2],
    'max_depth': [5, 10, 30, 50, 70, 90, 100, None],
}

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_predict = dt_model.predict(X_test)
init_acc = accuracy_score(y_test, y_predict)
print("Tacnost za Decision Tree pre podesavanja hiperparametara je: ",
      init_acc)

grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
grid_search_dt.fit(X_train, y_train)
print("Najbolji parametri za Decision Tree su: ", grid_search_dt.best_params_)
dt_model.criterion = grid_search_dt.best_params_["criterion"]
dt_model.max_depth = grid_search_dt.best_params_["max_depth"]
dt_model.ccp_alpha = grid_search_dt.best_params_[
    "ccp_alpha"]
dt_model.fit(X_train, y_train)
y_predict = dt_model.predict(X_test)
final_acc = accuracy_score(y_test, y_predict)
print("Tacnost za Decision Tree posle podesavanja hiperparametara je: ",
      final_acc)

if final_acc > init_acc:
    print("Podesavanje hiperparametara za Decision Tree je uspesno")

# Najbolji parametri za Decision Tree su:  {'ccp_alpha': 0, 'criterion': 'entropy', 'max_depth': None}
# Tacnost za Decision Tree posle podesavanja hiperparametara je:  0.9473360024638128

# Zakomentarisano je jer podesavanje traje po 20 min
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
# y_predict = rf_model.predict(X_test)
# init_acc = accuracy_score(y_test, y_predict)
# print("Tacnost za Random Forest pre podesavanja hiperparametara je: ",
#       init_acc)

# grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5)
# grid_search_rf.fit(X_train, y_train)
# print("Najbolji parametri za Random Forest su: ", grid_search_rf.best_params_)
# rf_model.n_estimators = grid_search_rf.best_params_["n_estimators"]
# rf_model.max_depth = grid_search_rf.best_params_["max_depth"]
# rf_model.min_samples_split = grid_search_rf.best_params_["min_sample_split"]

# rf_model.fit(X_train, y_train)
# y_predict = rf_model.predict(X_test)
# final_acc = accuracy_score(y_test, y_predict)
# print("Tacnost za Random Forest posle podesavanja hiperparametara je: ",
#       final_acc)

# if final_acc > init_acc:
#     print("Podesavanje hiperparametara za Random Forest je uspesno")
# Podesavanje parametara neuspesno (predugo traje)

# Odabir najbitnijih atributa

# Klasifikator sa podešenim hiperparametrima
# rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model = RandomForestClassifier()

# Treniranje modela na celom skupu podataka
rf_model.fit(X_train, y_train)

# Izračunavanje značajnosti atributa
feature_importance = rf_model.feature_importances_

# Sortiranje atributa prema značajnosti
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_features = X_train.columns[sorted_indices]
sorted_importance = feature_importance[sorted_indices]

# Prikazivanje značajnosti atributa u obliku grafa
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importance)),
        sorted_importance, tick_label=sorted_features, color='green')
plt.xticks(rotation=90)
plt.xlabel('Atributi')
plt.ylabel('Značajnost')
plt.title('Značajnost atributa u Random Forest modelu')
plt.show()


# Izbacivanje manje znacajnih atributa
y_train = train_data["satisfaction"]
X_train = train_data.drop(
    ["satisfaction", "Gender_Male", "Gender_Female", "Food and drink"], axis=1)
y_test = test_data["satisfaction"]
X_test = test_data.drop(
    ["satisfaction", "Gender_Male", "Gender_Female", "Food and drink"], axis=1)

rf_model.fit(X_train, y_train)
y_predict = rf_model.predict(X_test)
print("Tacnost modela nakon izbacivanja manje znacajnih atributa: ",
      accuracy_score(y_test, y_predict))
