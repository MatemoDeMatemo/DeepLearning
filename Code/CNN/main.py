import torch
from torch.utils.data import DataLoader, random_split
from to_tensor import Tensors_Creator
from train import Model_Train


# === README ===
#
# === Skrypt ===
#  - kod sluzacy do treningu i zapisania modelu
#  - wymaga funkcji "Tensors_Creator"

# === Input ===
# - funkcja "Tensors_Creator" - tworzaca tensory ze zdjec oraz pliku json.

# === Output ===

# === FLOW ===
# - 1) przygotowanie sciezek
# - 2) zastosowanie funkcji "Tensors_Creator" - tworzacej tensory ze zdjec oraz pliku json. Tworzy: "Tensor_Data"
# - 3) rozbicie 'Tensor_Data' na test, train i validation data
# - 4) ustawienia modelu
# - 5) trening modelu
# - 6) zapisanie modelu
# - 7) zapis checkpointu - zapisuje statystyki modelu


#######################################################################################################################
#### === 1) Przygotowanie sciezek


################################ !! START PRACY STUDENTA !! ##########################################
### Przyklad z zajec z Kostkami (3 klasy)


Path_Images      = "Input\Foto_input"               # sciezka do sklasyfikowanych zdjec
Path_Annotations = r"Input\instances_default.json"  # sciezka do pliku json z opisem sklasyfikowanych zdjec (do pobrania z CVAT)

NUM_Classes = 3     # ilosc klas. Kazdy moze miec inna liczbe! Tlo to tez klasa
NUM_epochs = 10      # ilosc epok.
BATCH_size = 2     # wielkosc batch'y

merge_map= {1: 1, 2: 2} # Wazne! Sluzy do mozliwosci grupowania klas. Np. merge_map= {1: 1, 2: 1}, zamienia 2 klasy kostek na 1 klase "kostka".
                        # Wazne by to dobrze ustawic. Jesli macie np. 4 klasy (jedna to tlo) to ustawienie to: merge_map= {1: 1, 2: 2, 3: 3} itd.


################################ !! KONIEC PRACY STUDENTA !! ##########################################


#######################################################################################################################
#### === 2) zastosowanie funkcji "Tensors_Creator"

# === Przygotowanie funkcji ===

Visualisation= False
#merge_map= {1: 1, 2: 2}

# Funkcja ktora tworzy tensory ze zdjec i dodaje do nich ramki z annotacja
Tensor_Data = Tensors_Creator(Path_Images= Path_Images,
                              Path_Annotations= Path_Annotations,
                              Visualisation= Visualisation,
                              merge_map= merge_map )


#######################################################################################################################
#### === 3) rozbicie 'Tensor_Data' na test, train i validation data

def collate_fn(batch):
    return tuple(zip(*batch))

# Proporcje splitu
train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1

dataset_size = len(Tensor_Data)
train_size = int(train_ratio * dataset_size)
val_size   = int(val_ratio   * dataset_size)
test_size  = dataset_size - train_size - val_size  # domyka do całości

# Powtarzalny losowy split (seed)
g = torch.Generator().manual_seed(42)
train_ds, val_ds, test_ds = random_split(Tensor_Data, [train_size, val_size, test_size], generator=g)

# DataLoadery
train_loader = DataLoader(train_ds, batch_size= BATCH_size, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size= BATCH_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size= BATCH_size, shuffle=False, collate_fn=collate_fn)

print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")


# #######################################################################################################################
# #### === 7) zapis checkpointu (do wznowienia treningu)

model, device = Model_Train(train_loader = train_loader,
            val_loader =val_loader,
            test_loader = test_loader,
            NUM_CLASSES = NUM_Classes,
            num_epochs = NUM_epochs,
            Save_The_Whole_Model = True,
            Save_Models_architecture = False,
            Save_Checkpoint = False)

