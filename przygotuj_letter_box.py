import os
from PIL import Image

def process_images(input_base_dir, output_base_dir):
    """
    Przechodzi przez foldery, znajduje obrazy i tworzy ich kwadratowe
    wersje z czarnymi pasami (letterboxing).
    """
    print(f"Przetwarzam obrazy z: {input_base_dir}")
    print(f"Zapisuję do: {output_base_dir}")
    
    licznik = 0
    # os.walk "spaceruje" po wszystkich podfolderach
    for root, dirs, files in os.walk(input_base_dir):
        for plik in files:
            # Interesują nas tylko pliki graficzne
            if not plik.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # 1. Otwórz oryginalny plik
            sciezka_pliku = os.path.join(root, plik)
            try:
                with Image.open(sciezka_pliku) as img:
                    # Konwertuj na RGB (dla spójności, usuwa np. kanał alpha z PNG)
                    img = img.convert('RGB')
                    
                    # 2. Znajdź dłuższy bok, aby stworzyć kwadrat
                    width, height = img.size
                    max_dim = max(width, height)
                    
                    # 3. Stwórz nowe, czarne, kwadratowe tło
                    new_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
                    
                    # 4. Oblicz pozycję, aby wycentrować oryginalny obraz
                    paste_x = (max_dim - width) // 2
                    paste_y = (max_dim - height) // 2
                    
                    # 5. Wklej oryginalny obraz na czarne tło
                    new_img.paste(img, (paste_x, paste_y))
                    
                    # 6. Przygotuj ścieżkę zapisu z zachowaniem podfolderów
                    # Oblicz ścieżkę względną (np. "Dzik" lub "Sarna")
                    relative_path = os.path.relpath(root, input_base_dir)
                    # Stwórz folder wyjściowy (np. .../train_letterboxed/Dzik)
                    output_subfolder = os.path.join(output_base_dir, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)
                    
                    # Zapisz nowy plik
                    save_path = os.path.join(output_subfolder, plik)
                    new_img.save(save_path)
                    
                    licznik += 1
                    if licznik % 100 == 0:
                        print(f"Przetworzono {licznik} obrazów...")

            except Exception as e:
                print(f"BŁĄD: Pominąłem plik {sciezka_pliku}. Błąd: {e}")
                
    print(f"Zakończono. Łącznie przetworzono {licznik} obrazów.\n")

# --- GŁÓWNA CZĘŚĆ SKRYPTU ---

# 1. Znajdź folder, z którego uruchomiono skrypt
base_dir = os.getcwd() 
print(f"Folder roboczy: {base_dir}")

# 2. Zdefiniuj ścieżki WEJŚCIOWE
train_in_dir = os.path.join(base_dir, "OpenAnimalTracks", "cropped_imgs", "train")
test_in_dir = os.path.join(base_dir, "OpenAnimalTracks", "cropped_imgs", "test")

# 3. Zdefiniuj ścieżki WYJŚCIOWE
train_out_dir = os.path.join(base_dir, "OpenAnimalTracks", "cropped_imgs", "train_letterboxed")
test_out_dir = os.path.join(base_dir, "OpenAnimalTracks", "cropped_imgs", "test_letterboxed")

# 4. Uruchom przetwarzanie
process_images(train_in_dir, train_out_dir)
process_images(test_in_dir, test_out_dir)

print("Wszystko gotowe!")
print("Teraz zaktualizuj ścieżki 'TRAIN_DIR' i 'TEST_DIR' w swoim skrypcie 'apps.py'.")
