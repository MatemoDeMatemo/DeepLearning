import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === README ===
#
# === Skrypt ===
#  - Funkcja Tensors_Creator - kazda obiekt zawiera pare - tensor + boxy z klasami
#  - Zwracany jest obiekt Tensor_Ready, zawierajacy zlozenie tensorow obrazow wraz z klasami i boxami
#  - Wyswietlane sa metadane zdjec
#  - Wyswietlane jest wybrane zdjecie po numerze Visualisation wraz z boxami

# === CONFIG === # Turn on/off
Run_Function = False

Path_Images      = r"Dice_Test\Data\Images"
Path_Annotations = r"Dice_Test\Data\Coco\instances_default.json"
merge_map = {1: 2, 2: 6}  # np. scalenie klas 1 i 2 w jedną
Visualisation = 1 # or Visualisation = False





# ######################################### == Function Start == ######################################################
#######################################################################################################################
#### === 0) Def Funkcji ====

def Tensors_Creator(Path_Images, Path_Annotations, Visualisation = False, merge_map = {}):


#######################################################################################################################
#### === 1) Wyciagnij metadane ====

    print("Pliki w folderze ze zdjęciami:")
    print(os.listdir(Path_Images))

    # === Wczytaj dane COCO ===
    coco_data = COCO(Path_Annotations)
    coco_Cat_ID = coco_data.getCatIds()
    coco_IMG_ID = coco_data.getImgIds()
    coco_Ann_ID = coco_data.getAnnIds()

    # Wypisz podstawowe dane
    print("\nPlik json:")
    print("  Liczba klas:     ", len(coco_Cat_ID))
    print("  Liczba obrazów:  ", len(coco_IMG_ID))
    print("  Liczba anotacji: ", len(coco_Ann_ID))

    # Wypisz nazwy klas
    categories = coco_data.loadCats(coco_Cat_ID)
    print("\nNazwy klas w pliku COCO:")
    for cat in categories:
        print(f"  ID {cat['id']}: {cat['name']}")


#######################################################################################################################
#### === 2) Stworz klase ktora przetwarza anotacje i zdjecia na jednego tensora z klasami i boxami ====

    if merge_map == {}: return

    # === DATASET ===
    class CocoMergeDataset(CocoDetection):
        def __init__(self, root, annFile, merge_map=None, transform=None):
            super().__init__(root, annFile, transform)
            self.merge_map = merge_map or {}

        def __getitem__(self, idx):
            img, target = super().__getitem__(idx)

            boxes, labels = [], []              # stworz puste obiekty

            for ann in target:
                boxes.append(ann["bbox"])       # dodaj wierzcholki prostokata
                label_id = ann["category_id"]   # dodaj klase

                # Zastosuj mapowanie scalające (jeśli jest)
                if label_id in self.merge_map:
                    label_id = self.merge_map[label_id]

                labels.append(label_id)

            # Jeśli brak anotacji — tworzymy poprawne puste tensory
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                boxes[:, 2:] += boxes[:, :2]  # [x, y, w, h] -> [x1, y1, x2, y2]
                labels = torch.tensor(labels, dtype=torch.int64)

            return T.ToTensor()(img), {"boxes": boxes, "labels": labels} # zwroc tensor oraz prostokaty wraz z klasami


    ### Przygotuj tensory ###
    Tensor_Ready = CocoMergeDataset(
        root=Path_Images,
        annFile=Path_Annotations,
        merge_map=merge_map
    )


    img, target = Tensor_Ready[Visualisation]  # tylko jeden przykład
    print("\nAdnotacje dla przykładowego obrazu:")
    print("Numer obrazu: ", int(Visualisation))
    print(target)


#######################################################################################################################
#### === 3) Wizualizacja wybranego zdjecia z boxami ====


    if Visualisation > 0:

        # === Wizualizacja ===
        img, target = Tensor_Ready[Visualisation]  # Visualisation jak liczbowa to wybiera numer zdjecia do wyswietlenia
        img_np = img.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_np)

        boxes = target["boxes"]
        labels = target["labels"]

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            width, height = x2 - x1, y2 - y1

            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

            class_name = str(labels[i].item())

            ax.text(
                x1, y1 - 5, class_name,
                color='yellow', fontsize=12, weight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=2)
            )

        plt.axis("off")
        plt.show()

    print("Ilosc tensorow to: ", len(Tensor_Ready))
    print("Rozmiar tensorow to: ", img.shape)


#######################################################################################################################
#### === 4) Output ====

    return Tensor_Ready

################################################ == END == ############################################################

if Run_Function == True:
    Tensors_Creator(Path_Images, Path_Annotations, Visualisation, merge_map)