#### 1] Konstrukcja Perceptronu ####

# Dane wejsciowe: 
input   = [1, 0, 1]
wagi    = [5, 4, 3]

bias    = -10

# Definicja funkcji aktywacji - funkcji progowej 
def Funkcja_Progowa(suma):
    if suma + bias >= 0 :
        return 1
    else:
        return 0

# Policz wynik dla wybranych danych wejsciowych, wag i bias'u

suma = 0

for i in range(0, len(wagi)):
    print(wagi[i])
    suma += input[i] * wagi[i]

output = Funkcja_Progowa(suma)



#### 2] Trenowanie Perceptronu ####


# Dane treningowe: [wejscia, oczekiwany wynik]
dane_treningowe = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

dane_treningowe = [
    ([1, 0.5],  0),
    ([0.5, 1],  0),
    ([1, 2],    0),
    ([1.5, 2],  0),
    ([2, 3],    0),
    
    ([2, 0.5],  1),
    ([3, 1],    1),
    ([4, 1],    1),
    ([3.5, 2],  1),
    ([4, 3],    1),
]

wagi  = [0.0, 0.0]  # zaczynamy od zerowych wag
bias  = 0.0
wspolczynnik_uczenia = 0.1
epoki = 10

print("\n=== Nauka perceptronu ===")

for epoka in range(epoki):
    bledy = 0
    print(f"\n-- Epoka {epoka + 1} --")

    for dane_wejsciowe, oczekiwany in dane_treningowe:

        # Oblicz sume
        suma = 0
        for i in range(len(wagi)):
            suma += dane_wejsciowe[i] * wagi[i]

        # Oblicz output
        output = Funkcja_Progowa(suma)

        # Oblicz blad
        blad = oczekiwany - output

        if blad != 0:
            bledy += 1
            print(f"  Wejscie: {dane_wejsciowe} | Oczekiwano: {oczekiwany} | Otrzymano: {output} | Błąd: {blad}")

            # Aktualizuj wagi
            for i in range(len(wagi)):
                wagi[i] += wspolczynnik_uczenia * blad * dane_wejsciowe[i]
            bias += wspolczynnik_uczenia * blad

            print(f"  Nowe wagi: {[round(w, 2) for w in wagi]} | Nowy bias: {round(bias, 2)}")
        else:
            print(f"  Wejscie: {dane_wejsciowe} | Oczekiwano: {oczekiwany} | Otrzymano: {output} | OK ✓")

    if bledy == 0:
        print(f"\nPerceptron nauczył się po {epoka + 1} epoce!")
        break

print(f"\nOstateczne wagi: {[round(w, 2) for w in wagi]}")
print(f"Ostateczny bias: {round(bias, 2)}")
    
