import sys
import os

print("=" * 60)
print("   PANDUAN SETUP & VERIFIKASI LINGKUNGAN GPU (LOCAL SYSTEM)")
print("=" * 60)

# 1. Cek versi Python
print(f"Versi Python: {sys.version}")

# 2. Cek PyTorch & CUDA
try:
    import torch
    print("PyTorch: TERINSTALL (OK)")
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA (Akselerasi GPU): TERSEDIA (OK) [OK]")
        print(f"Model GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA (Akselerasi GPU): TIDAK TERSEDIA [X] (Berjalan di CPU)")
        print("\n--- Panduan Mengaktifkan GPU di Komputer Lokal ---")
        print("Jika komputer Anda memiliki kartu grafis NVIDIA, ikuti langkah berikut untuk mengaktifkan CUDA:")
        print("1. Hapus instalasi PyTorch CPU bawaan:")
        print("   pip uninstall torch torchvision torchaudio -y")
        print("2. Kunjungi situs resmi PyTorch (https://pytorch.org/get-started/locally/)")
        print("3. Pilih konfigurasi OS Anda dan CUDA version (misalnya CUDA 11.8 atau CUDA 12.1)")
        print("4. Jalankan perintah instalasi yang direkomendasikan, contoh:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("PyTorch: TIDAK TERINSTALL [X]")
    print("Silakan install PyTorch terlebih dahulu dengan perintah:")
    print("pip install torch")

# 3. Cek dependensi lainnya
packages = ['pandas', 'numpy', 'sentence_transformers', 'sklearn', 'gradio', 'tqdm']
print("\n--- Status Dependensi Lainnya ---")
for pkg in packages:
    try:
        import_name = pkg
        if pkg == 'sklearn':
            import_name = 'sklearn'
        elif pkg == 'sentence_transformers':
            import_name = 'sentence_transformers'
            
        __import__(import_name)
        print(f"- {pkg:21}: TERINSTALL [OK]")
    except ImportError:
        print(f"- {pkg:21}: BELUM TERINSTALL [X]")

print("\nTips Menjalankan di Google Colab:")
print("Notebook ini dapat dijalankan langsung di Google Colab dengan mengunggah file .ipynb tersebut.")
print("Untuk menggunakan GPU di Colab, pilih menu: 'Runtime' -> 'Change runtime type' -> 'T4 GPU' -> 'Save'.")
print("=" * 60)
