
import pandas as pd # do tworzenia data frame
from sklearn.model_selection import train_test_split # pip install scikit-learn
from sklearn.preprocessing import StandardScaler # do skalowania danych
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

start_time = time.time()

########################################################################################################################################################################
####################################################################### 1] Wczytywanie danych z csv ####################################################################

#### Wczytanie i obrobka danych
## Import the data
df = pd.read_csv('AmesHousing.csv',
                 sep=';',  # separator (domyślnie ',', czasem ';' lub '\t')
                 decimal='.',  # separator dziesiętny
                 encoding='utf-8',  # kodowanie (ważne przy polskich znakach!)
                 )

## View the data
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())


# Wybierz cechy
X = df[['Lot Area', 'Year Built', 'Gr Liv Area', 'Neighborhood']]
y = df['SalePrice']

## Poprawka danych

X = pd.get_dummies(X, columns=['Neighborhood'])



## Podział na dane treningowe, walidacyjne i testowe.

#### Podziel dane na treningowe oraz chwilowe (temporal)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Dane chwilowe podziel na validacyjne i testowe - podziel po polowie
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Jaki mamy rozmiar inputu?
input_size = X_train.shape[1]

## Standaryzacja
# standaryzujemy po podzialem bo dane z testu "przeciekłyby" do skalera – data leakage

scaler = StandardScaler()  # Przekształca każdą kolumnę tak, żeby miała średnią = 0 i odchylenie standardowe = 1:

scaler.fit(X_train)        # "zapamiętaj średnią i odchylenie z treningu"

X_train = scaler.transform(X_train)  # zastosuj to na treningu
X_val   = scaler.transform(X_val)    # zastosuj TE SAME wartości na walidacji
X_test  = scaler.transform(X_test)   # zastosuj TE SAME wartości na teście

## funkcja scaler zwraca obiekty typu numpy
# Zamieniamy dane numpy na tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)
# unsqueeze(1) zamienia shape (N,) na (N,1) – wymagane przez sieć  -- ? by zamiast wektora danych miec macierz o 1 kolumnie.

print(f"Train: {X_train_t.shape}, Val: {X_val_t.shape}, Test: {X_test_t.shape}")


########################################################################################################################################################################
####################################################################### 2] Architektura modelu #########################################################################

#### Deklaracja modelu sieci

# Deklarujemy klase w oparciu o szkielet nn. module.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)    		# pierwsza warstwa
        self.fc2 = nn.Linear(64, 32)    	# druga warstwa
        self.fc3 = nn.Linear(32, 1)     	# warstwa wynikowa
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # ostatnia warstwa bez aktywacji!
        return x

model = NeuralNetwork()
print(model)


########################################################################################################################################################################
####################################################################### 3] Trening modelu ##############################################################################

#### Parametry treningu

# Funkcja straty
loss_fn = nn.MSELoss()  # Mean Squared Error
#loss_fn = nn.L1Loss()   # Mean Absolute Error (MAE)

# Learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Jak analizowac dane z batchy?
# Łączymy X i y w jeden zbiór, by dane pasowaly do siebie w batchach
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)

# DataLoader automatycznie kroi na batche i tasuje dane
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Przed kazda epoka losuj zawartosc batchy.
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False) # nie potrzeba mieszac


#### Trening

# Ilosc epok
epochs = 100

for epoch in range(epochs):

    ## Trening
    model.train()  # tryb treningu
    train_loss = 0

    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)           # 1] liczy przewidywania dla batcha forward pass – sieć liczy predykcję
        loss = loss_fn(y_pred, y_batch)   # 2] liczymy blad liczymy błąd

        optimizer.zero_grad()             # zerujemy gradienty z poprzedniego batcha - po kazdym batchu gradienty są zsumowane
        loss.backward()                   # liczymy pochodne bledu wzgledem wag. backward pass – liczymy gradienty
        optimizer.step()                  # aktualizujemy wagi

        train_loss += loss.item()         # zbieramy błąd batcha

    train_loss /= len(train_loader)       # średni błąd epoki

    ## Walidacja
    model.eval()   # tryb ewaluacji (wyłącza dropout itp.)
    val_loss = 0

    with torch.no_grad():                 # nie liczymy gradientów – oszczędność pamięci
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            val_loss += loss_fn(y_pred, y_batch).item()

    val_loss /= len(val_loader) # jaki był sredni loss epoki? Suma lossow baczy przez liczbe batchy.

    ## Wypisz co 10 epok
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")

# Train loss - jak model radzi sobie na danych treningowych.
# Val loss - jak model radzi sobie na nowych danych




########################################################################################################################################################################
####################################################################### 4] Test modelu #################################################################################

#### Test modelu ####
model.eval()

with torch.no_grad():
    y_pred_test = model(X_test_t)
    test_loss = loss_fn(y_pred_test, y_test_t)

print(f"Test Loss: {test_loss.item():.2f}")



y_true = y_test_t.numpy()
y_pred = y_pred_test.numpy()

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print(f"MAE:  {mae:.0f} $")
print(f"RMSE: {rmse:.0f} $")
print(f"R²:   {r2:.4f}") # ile procent zmiennosci tlumaczy nasz model


end_time = time.time() # wypisz czas działania modelu

print(f"Czas treningu: {end_time - start_time:.2f} sekund")


########################################################################################################################################################################
####################################################################### 5] Zapis modelu ################################################################################

#### Zapis modelu - 2 sposoby
# Cały model
torch.save(model, 'model_full.pth') # zapis referencji klasy

# Zapis tylko wag i biasów, bez architektur - preferowany
torch.save(model.state_dict(), 'model_parametry.pth')

# Zapis parametrów skalera - bo trzeba tak samo przeskalowac dane
joblib.dump(scaler, 'scaler.pkl')   # zapis
scaler = joblib.load('scaler.pkl')  # wczytanie

# Wypisz ile kolumn ma input
print(input_size)
feature_columns = X.columns
joblib.dump(feature_columns, "columns.pkl")

print(X.columns)