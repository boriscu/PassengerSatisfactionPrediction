import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Tacka 1: Početno preprocesiranje podataka (proveriti da li postoje nedostajuće vrednosti ili anomalije, pretvoriti stringove u brojeve, izbaciti neke atribute ako je očigledno da ne utiču na formiranje izlaza i sl.)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Izbor numeričkih kolona
numeric_columns = ['Age', 'Flight Distance',
                   'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# Pregled podataka
# train_data.info()
# test_data.info()

# Prve dve kolone su beskorisne (Unamed, id)
train_data = train_data.drop(train_data.iloc[:, [0, 1]], axis=1)
test_data = test_data.drop(test_data.iloc[:, [0, 1]], axis=1)

# Provera balansiranosti skupa - Vidimo da nam je skup poprilicno izbalansiran
plt.pie(train_data.satisfaction.value_counts(), labels=[
        "Neutral or dissatisfied", "Satisfied"], colors=sns.color_palette("YlOrBr"), autopct='%1.1f%%')
plt.show()
# Provera nedostajucih vrednosti
# print(train_data.isnull().sum())
# print("#######################")
# print(test_data.isnull().sum())

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
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelaciona matrica numeričkih promenljivih')
plt.show()
# Scatter plot prikaz zavisnosti dve kolone
plt.scatter(train_data['Arrival Delay in Minutes'],
            train_data['Departure Delay in Minutes'], alpha=0.5)
plt.show()

# Nakon primecivanja jake korelacije izmedju arrival i departure delay dropujemo arrival delay
train_data = train_data.drop('Arrival Delay in Minutes', axis=1)
test_data = test_data.drop('Arrival Delay in Minutes', axis=1)

# Update numerickih kolona
numeric_columns = ['Age', 'Flight Distance',
                   'Departure Delay in Minutes']

# Tacka 2 : Eksplorativna analiza skupa (proveriti da li postoje neke jake korelacije, prikazati na grafiku raspodelu ciljnog atributa u odnosu na neke nezavisne promenljive i sl.)

# Vecina putnika koji su leteli economy ili economy plus klasom su bili nezadovoljni, a biznis su zadovoljni
sns.countplot(x='Class', hue='satisfaction', palette="YlOrBr", data=train_data)
plt.show()

# Svi koji su rejtovali wifi sa  5 zvezdica su zadovoljni
sns.countplot(x='Inflight wifi service', hue='satisfaction',
              palette="YlOrBr", data=train_data)
plt.show()

# Procenat zadovoljstva putnika po polu
gender_counts = train_data['Gender'].value_counts()
satisfaction_counts = train_data['satisfaction'].value_counts()

satisfied_gender_counts = train_data[train_data['satisfaction']
                                     == 'satisfied']['Gender'].value_counts()

satisfied_percentages = satisfied_gender_counts / gender_counts * 100

plt.bar(satisfied_percentages.index,
        satisfied_percentages.values, color=['blue', 'red'])
plt.title('Procenat zadovoljstva putnika prema polu')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.ylim(0, 100)
plt.show()

# Zadovoljstvo i godine
fig, axes = plt.subplots(3, 1, figsize=(8, 15))

# Zadovoljstvo putnika po godinama
age_satisfaction = train_data.groupby('Age')['satisfaction'].value_counts(
    normalize=True).unstack().reset_index()

axes[0].plot(age_satisfaction['Age'],
             age_satisfaction['satisfied'] * 100, label='Satisfied')
axes[0].set_title('Zadovoljstvo putnika u odnosu na starost')
axes[0].set_xlabel('Starost')
axes[0].set_ylabel('Procenat zadovoljstva')
axes[0].legend()

# Ocene "Leg room service" po godinama
age_legroom = train_data.groupby(
    'Age')['Leg room service'].mean().reset_index()

axes[1].plot(age_legroom['Age'], age_legroom['Leg room service'])
axes[1].set_title('Ocene "Leg room service" u odnosu na starost')
axes[1].set_xlabel('Starost')
axes[1].set_ylabel('Ocena "Leg room service"')

# Ocene "Inflight wifi service" po godinama
age_wifi = train_data.groupby(
    'Age')['Inflight wifi service'].mean().reset_index()

axes[2].plot(age_wifi['Age'], age_wifi['Inflight wifi service'])
axes[2].set_title('Ocene "Inflight wifi service" u odnosu na starost')
axes[2].set_xlabel('Starost')
axes[2].set_ylabel('Ocena "Inflight wifi service"')

plt.tight_layout()  # Za bolji raspored subplotova
plt.show()

# Raspodela klasa letenja po godinama
age_class = train_data.groupby(['Age', 'Class']).size().unstack()

age_class.plot(kind='bar', stacked=True)
plt.title('Raspodela klasa letenja u odnosu na starost')
plt.xlabel('Starost')
plt.ylabel('Broj putnika')
plt.legend(title='Klasa letenja')
plt.show()


# Broj lojalnih putnika po godinama
sns.histplot(train_data, x="Age", hue="Customer Type",
             multiple="stack", palette="YlOrBr", edgecolor=".3", linewidth=.5)
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

# Primeti se velik broj anomalija u departure delay pa nju transformisemo
train_data['Departure Delay in Minutes'] = np.log(
    train_data['Departure Delay in Minutes'] + 1)


test_data['Departure Delay in Minutes'] = np.log(
    test_data['Departure Delay in Minutes'] + 1)

# Ponovno racunanje anomalija za kriticnu kolonu
z_scores = np.abs(stats.zscore(train_data['Departure Delay in Minutes']))
anomalies = (z_scores > 3)
print(
    f"Number of anomalies in {'Departure Delay in Minutes'}: {anomalies.sum()}")
