import torch
import torch.nn as nn
import pandas as pd
import joblib


#### Ile kolumn / ile inputu?
columns = joblib.load("columns.pkl")
print(len(columns))
input_size = len(columns)

# Przywolujemy stworzona architekture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)     # pierwsza warstwa
        self.fc2 = nn.Linear(64, 32)    # druga warstwa
        self.fc3 = nn.Linear(32, 1)     # warstwa wynikowa
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # ostatnia warstwa bez aktywacji!
        return x

model = NeuralNetwork()

#### Wczytaj model - wybierz jeden z 2 sposobów

## Model full

model = torch.load('model_full.pth', weights_only=False)
model.eval() # przelacza model w tryb "pracy" nie "uczenia"

# # Model parameters - LEPSZE

#
#
#
# model = NeuralNetwork()          # najpierw musisz odtworzyć architekturę - musi byc IDENTYCZNA
# model.load_state_dict(torch.load('model_parametry.pth'))
# model.eval()

#### Predykcja

import joblib
import torch

scaler = joblib.load('scaler.pkl')

nowe_dane = pd.DataFrame(columns=columns)
nowe_dane.loc[0] = 0

nowe_dane.loc[0, 'Lot Area'] = 8000
nowe_dane.loc[0, 'Year Built'] = 2000
nowe_dane.loc[0, 'Gr Liv Area'] = 1500


nowe_dane_scaled = scaler.transform(nowe_dane)  # ta sama normalizacja co podczas treningu!
nowe_dane_tensor = torch.tensor(nowe_dane_scaled, dtype=torch.float32)

with torch.no_grad():
    cena = model(nowe_dane_tensor)

print(f"Przewidywana cena: ${cena.item():,.0f}")