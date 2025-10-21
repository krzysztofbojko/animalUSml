import os
from PIL import Image, ImageOps 

# --- KONFIGURACJA ---
# Skrypt automatycznie wykryje folder, z którego jest uruchamiany
base_dir = os.getcwd() 
print(f"Wykryto folder roboczy: {base_dir}")

# Budujemy ścieżkę do folderu 'train' niezależnie od systemu (Win/Linux)
# os.path.join sam doda odpowiednie ukośniki (\ lub /)
GLOWNY_FOLDER = os.path.join(
    base_dir, 
    "OpenAnimalTracks", 
    "cropped_imgs", 
    "train"
)
# --- KONIEC KONFIGURACJI ---

print(f"Rozpoczynam pracę w folderze: {GLOWNY_FOLDER}")

# Sprawdzamy, czy folder istnieje, zanim zaczniemy
if not os.path.isdir(GLOWNY_FOLDER):
    print(f"BŁĄD: Nie mogę znaleźć folderu! {GLOWNY_FOLDER}")
    print("Upewnij się, że uruchamiasz skrypt z folderu 'ml-animal'.")
    exit() # Zakończ skrypt, jeśli ścieżka jest zła

licznik_plikow = 0

# os.walk() "spaceruje" po wszystkich podfolderach w GLOWNY_FOLDER
for root, dirs, files in os.walk(GLOWNY_FOLDER):
    for plik in files:
        
        # Sprawdzamy, czy plik to JPG i czy NIE JEST już kopią (aug_)
        if plik.lower().endswith(('.jpg', '.jpeg')) and not plik.lower().startswith('aug_'):
            
            # Tworzymy pełną ścieżkę do pliku
            sciezka_pliku = os.path.join(root, plik)
            # Dzielimy ścieżkę, aby dostać samą nazwę bez rozszerzenia
            nazwa_bez_rozszerzenia = os.path.splitext(sciezka_pliku)[0]
            
            try:
                # Otwieramy oryginalny obraz
                with Image.open(sciezka_pliku) as obraz_oryginalny:
                    
                    # Automatycznie korygujemy orientację zdjęcia (np. z telefonu)
                    obraz = ImageOps.exif_transpose(obraz_oryginalny)

                    # --- 1. Obrót 90 stopni ---
                    obraz_90 = obraz.rotate(90, expand=True)
                    obraz_90.save(f"{nazwa_bez_rozszerzenia}_aug_obrot90.jpg")

                    # --- 2. Obrót 180 stopni ---
                    obraz_180 = obraz.rotate(180)
                    obraz_180.save(f"{nazwa_bez_rozszerzenia}_aug_obrot180.jpg")

                    # --- 3. Obrót 270 stopni ---
                    obraz_270 = obraz.rotate(270, expand=True)
                    obraz_270.save(f"{nazwa_bez_rozszerzenia}_aug_obrot270.jpg")

                    # --- 4. Odbicie lustrzane (w poziomie) ---
                    obraz_lustro = obraz.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    obraz_lustro.save(f"{nazwa_bez_rozszerzenia}_aug_lustro.jpg")
                
                licznik_plikow += 1
                if licznik_plikow % 50 == 0:
                    print(f"Przetworzono {licznik_plikow} plików...")

            except Exception as e:
                # Wyświetlamy błąd, jeśli plik jest uszkodzony, ale idziemy dalej
                print(f"BŁĄD: Nie można przetworzyć pliku {sciezka_pliku}. Błąd: {e}")

print(f"\nGotowe! Stworzono 4 kopie dla {licznik_plikow} oryginalnych plików.")