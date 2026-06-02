Konstrukacja kodu:

Składa się z 4 plików .py:

- main.py [główny kod wykonujący. Korzysta z funkcji tworzącej tensory oraz trenującej model zawartych w plikach poniżej]
	- to_tensor.py [zawiera funkcję zamieniającą obraz na tensor, wraz z informacją o położeniu obiektów wraz z klasami]
	- train.py [zawiera funkcję do trenowania modelu. Zapisuje model w folderze projektu]

- predict.py [bierze zapisany model i przeprowadza predykcje na nowych zdjęciach. Tworzy excel ze zliczonymi obiektami]


1] - main.py
Input:

	Path_Annotations - ścieżka do pliku json z klasami obiektów. Plik jest pobieranylny ze strony CVAT. ["Input\instances_default.json"]
	Path_Images - ścieżka do pliku z oznaczonymi zdjęciami. ["Input\Foto_input"]

	
	Parametry do wybrania:
		NUM_Classes = 3		- [int], ile klas ma nasz problem? Pamiętajmy że jedną klasą jest tło. Więc w wypadku gdy model rozpoznaje 2 liczby na kostkach, mamy 3 klasy: background, 2 oczka, 6 oczek.
		NUM_epochs = 4		- [int], ile chcemy mieć epok
		BATCH_size = 1		- [int], jaki chcemy rozmiar batchy. 

		merge_map= {1: 1, 2: 2} # Wazne! Sluzy do mozliwosci grupowania klas. Np. merge_map= {1: 1, 2: 1}, zamienia 2 klasy kostek na 1 klase "kostka". 
                        		# Wazne by to dobrze ustawic. Jesli macie np. 4 klasy (jedna to tlo) to ustawienie to: merge_map= {1: 1, 2: 2, 3: 3} itd. 

Output: 

	fasterrcnn_model_{timestamp}.pth - plik typu .pth będący wytrenowanym modelem. 


2] predict.py
Input:

	fasterrcnn_model_{timestamp}.pth - ścieżka (nazwa) do wytrenowanego modelu. 
	Folder_To_Predict - ścieżka do folderu ze zdjęciami do klasyfikacji

	Parametry do wybrania:
		NUM_Classes = 3		- [int], ile klas ma nasz problem? Pamiętajmy że jedną klasą jest tło. Więc w wypadku gdy model rozpoznaje 2 liczby na kostkach, mamy 3 klasy: background, 2 oczka, 6 oczek.
		Threshold = 0.5		- [double - od 0 do 1] - od jakiego progu mamy predicty uznawać za właściwe?
		Visualisation = "All"   - [char lub int] - wskazujemy ile zdjęć z folderu chcemy zwizualizować - wszystkie czy tylko kilka pierwszych?

Output:
	
	plik excel z ilością obiektów na zdjęciach. Zapisuje się na pulpicie. 


##### 
Kroki do wykonania:

1] Pobrac ze strony cvat adnotacje w postaci COCO 1.0 (plik json)
2] Stworzyć w pyhonie projekt z plikami main.py, to_tensor.py, train.py, predict.py i wkleić odpowiednio kod. Stworzyć odpowiednio folder Input a w nim foldery Foto_input (na oznaczone zdjecia), fun_class (na zdjecia do predykcji)
3] Do folderu Input wrzucić plik json
4] Do folderu Input/Foto_input wgrać oznaczone zdjęcia
5] Do folderu Input/fun_class wrzucić zdjecia do oznaczenia (w tym te nietypowe)

Wewnątrz kodu
6) W main.py ustawić odpowiednio parametry: NUM_Classes, NUM_epochs, BATCH_size, merge_map (sciezki powinny byc dobre). SZCZEGOLNIE ważne jest dobre podanie ilości klas w NUM_Classes oraz merge_map
7) Uruchomić main.py. Zapisać wytrenowany model, jeśli będziemy zadowoleni z wyników. 

8) W predict.py ustawić odpowiednio parametry: Path_Model, NUM_Classes. 
9) Uruchomić predict.py. 

10) Wyciągnąć wnioski i zanotować ciekawe obserwacje :)

W razie trudności proszę o kontakt! Mój pokój to 1.26, mail to matdra@amu.edu.pl
