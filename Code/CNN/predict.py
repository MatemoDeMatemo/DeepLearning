import os
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


# === README ===
#
# === Skrypt ===
#  - wczytuje wytrenowany model Faster R-CNN z Pulpitu,
#  - wykonuje predykcje na wszystkich obrazach podanych w folderze na pulpicie,
#  - zapisuje wynik predykcji: ilosc obiektow na zdjeciu, nazwy obrazow, format oraz rozmiar
#  - wizualizuje wykryte obiekty (ramki + score) dla n obrazow lub wszystkich obrazow
#
# === Output ===
#  - funkcja "Model_Predict_And_Visualise" robiaca ponizsze:
#  - wydruk slownika prediction,
#  - wyswietlenie obrazu z naniesionymi ramkami.
#  - plik csv z wynikami analizy
#
# === FLOW ===
#  - 0) definicja funkcji
#  - 1) przygotowanie sciezek
#  - 2) wczytanie gotowego modelu
#  - 3) stworzenie df z metadanymi obrazow przy pomocy petli, sprawdzanie czy wszystkie pliki w folderze to zdjecia
#  - 4) rozpoczecie petli w ktorej na kazdym zdjeciu wykonywany jest "predict"; Zapis wyniku predykcji do df
#  - 5) kontyuacja petli w ktorej dla wybranej ilosci obrazow robiona jest wizualizacja wyniku predykcji
#  - 6) zapis df do pliku csv na pulpicie
#
# === CONFIG === Model_Predict_And_Visualise()
Run_Function = True # Zamien na True by uruchomic funkcje

###################################### !! PRZYKLAD !! ################################################

Path_Model = r"fasterrcnn_model_20260602_1743.pth"
Folder_To_Predict = r"Input\fun_class"
NUM_Classes = 3 # ilosc klas. Kazdy moze miec inna liczbe! Tlo to tez klasa
Threshold = 0.5
Visualisation = "All" # moze byc liczba lub "All"


################################ !! START PRACY STUDENTA !! ##########################################

Path_Model = r"fasterrcnn_model_....pth"
Folder_To_Predict = r"Input\fun_class"
NUM_Classes = ...  # ilosc klas. Kazdy moze miec inna liczbe! Tlo to tez klasa
Threshold = 0.5
Visualisation = "All" # moze byc liczba lub "All"

################################ !! KONIEC PRACY STUDENTA !! ##########################################




# ######################################### == Function Start == ######################################################
#######################################################################################################################
#### === 0) Def funkcji ====

def Model_Predict_And_Visualise(Model_FromDesktop, Folder_From_Desktop, NUM_CLASSES, Threshold = 0.5, Visualisation = "All"):
#def Model_Predict_And_Visualise():

    # Path_Model = "fasterrcnn_best.pth"
    # Folder_To_Predict = "Zdjecia_Model"
    # NUM_Classes = 2
    # Threshold = 0.5
    # Visualisation = "All" # moze byc liczba lub "All


#######################################################################################################################
#### === 1) Przygotowanie sciezek

    # Path_Desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    # Path_TrainedModel   = os.path.join(Path_Desktop, Path_Model)
    # Path_ImageFolder = os.path.join(Path_Desktop, Folder_To_Predict)


    Path_Desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    Path_TrainedModel = Path_Model
    Path_ImageFolder = Folder_To_Predict


#######################################################################################################################
#### === 2) Wczytanie modelu ====

    # --- Architektura jak w treningu ---
    # Uwaga: uzycie nowszego API - bez deprecated "pretrained"
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        NUM_Classes
    )

    # --- Wczytanie wag ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Uzywane urzadzenie:       ", device)

    state_dict = torch.load(Path_TrainedModel, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("Model wczytany i gotowy do PREDYKCJI.")


#######################################################################################################################
#### === 3) Stworzenie df na metadane obrazow w folderze, sprawdzenie

    # --- Wypisz obiekty w folderze z obrazami ---
    valid_ext = ('.jpg', '.jpeg', '.png')

    files = [
        f for f in os.listdir(Path_ImageFolder)
        if f.lower().endswith(valid_ext)
]

    # --- Stworz df na metadane i zapelnij go danymi przy pomocy petli ---
    data = []

    for f in files:
        name, ext = os.path.splitext(f)
        full_path = os.path.join(Path_ImageFolder, f)

        # Pobierz rozmiar tylko jesli to obraz
        try:
            with Image.open(full_path) as img:
                size = img.size   # (width, height)
        except Exception:
            size = None          # jesli plik nie jest obrazem lub uszkodzony

        # Polacz dane
        data.append({
            "filename": f,
            "extension": ext.lower(),
            "image_size": size
        })

    # --- przypisz dane do df ---
    df = pd.DataFrame(data)

    # --- podaj metadane folderu z obrazami ---
    print("Liczba elementow w folderze na zdjecia: ", len(df))
    print("Zawartosc folderu na zdjecia: \n", df)

    # --- sprawdzenie czy wszystkie pliki w tym samym rozszerzeniu ---
    if len(df["extension"].unique())!= 1:
        print("Rozne rodzaje rozszerzen plikow!: ", df["extension"].unique())
        #time.sleep(10)
    print(df["extension"].unique())


# #######################################################################################################################
# #### === 4) Wczytanie obrazu i predykcja ==== START LOOP

    # --- Stworz przyszla kolumne do zliczania obiektow w obrazie
    Col_Liczba_Obiektow = []

    # --- Ustal ile obrazow ma byc w kroku 5) zwizualizowanych
    if Visualisation == "All":
        Max_Visualisation = len(files)
    else: Max_Visualisation = Visualisation

    # --- Rozpocznij petle ---
    for names in files:
        print("\n\nNazwa obrazu: ", names)
        # Stworz sciezke do konkretnego obrazu
        Path_ImageToPredict = os.path.join(Path_ImageFolder, names)

        # Otworz obraz i zamien go na tensor
        image = Image.open(Path_ImageToPredict).convert("RGB")
        image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)

        with torch.no_grad(): # Polecenie by model nie sledzil i nie zapamietywal operacji
            prediction = model(image_tensor)

        print("\nSurowy wynik prediction:")
        print(prediction)

        # Policz ilosc obiektow na obrazie
        L_Obiekty_od_Treshold = (prediction[0]["scores"] > Threshold).sum().item()
        print("Ilosc obiektow na zdjeciu: ", L_Obiekty_od_Treshold)
        Col_Liczba_Obiektow.append(L_Obiekty_od_Treshold)


    # #######################################################################################################################
    # #### === 5) Wizualizacja wynikow ==== KONTYNUACJA LOOP & END

        #### Wizualizacja
        if Max_Visualisation > 0:

    # --- Pobierz wyniki z pierwszego (i jedynego) obrazu ---
            pred = prediction[0]
            boxes  = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()

            # Konwersja obrazu do numpy
            img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()

            fig, ax = plt.subplots(1, figsize=(12, 8)) # Podaj wymiary obrazu
            ax.imshow(img_np)

            # Naloz bounding boxy na obiekty
            for box, score, label in zip(boxes, scores, labels):
                if score < Threshold:
                    continue

                x1, y1, x2, y2 = box
                width  = x2 - x1
                height = y2 - y1

                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor='lime',
                    facecolor='none'
                )
                ax.add_patch(rect)

                ax.text(
                    x1, y1 - 5,
                    f"{label} ({score:.2f})",
                    color='yellow',
                    fontsize=12,
                    weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, pad=2)
                )

            # Wizualizacja obrazu
            plt.axis("off")
            plt.show()

            # Zmniejsz counter o 1 dla ilosci wizualizacji
            Max_Visualisation -= 1

    # Dodaj do df kolumne z iloscia obiektow wykrytych na obrazach
    df["object_count"] = Col_Liczba_Obiektow
    print(df)


    # #######################################################################################################################
    # #### === 5) Zapis df jako csv

    while True:
        answer = input("\nCzy chcesz zapisac wynik w pliku csv? \nWpisz: Tak/Nie: ").strip().lower()

        if answer in ("tak", "t"):
            csv_path = os.path.join(Path_Desktop, "Model_CNN_result.csv")
            df.to_csv(csv_path, index=False, sep=';', decimal=',')
            print("\nPlik csv zapisany na pulpicie")
            break
        elif answer in ("nie", "n"):
            break
        else:
            print("Invalid input. Please type Yes or No.")


# ################################################ == END == ############################################################

if Run_Function == True:
    Model_Predict_And_Visualise(Path_Model, Folder_To_Predict, NUM_Classes,  Threshold, Visualisation)