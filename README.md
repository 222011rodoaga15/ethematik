# ethematik
Bu Ethem ve Rodin tarafından geliştirilen, 2026 maarif model kapsamlı 9. 10. 11. ve 12. sınıflar için hazırlanmış yapay zeka ile destekli ders çalışma kaynağı.

# Dependicididi
Öncelikle kodun çalışması için gerekli kütüphaneleri pip install ile indirmeniz lazım. Gerekli olan malukatlar requirements.txt'de bulunuyor.

# Python kurulumu
Python'ı kurmak için işletim sisteminize göre aşağıdaki komutları kullanabilirsiniz.

## Bu arch'ıArch için:
sudo pacman -S python

## Debian/Ubuntu için:
sudo apt install python3 python3-pip

## Fedora için:
sudo dnf install python3 python3-pip

## Windows için python.org'dan indirip kurabilirsiniz. Kurulumda "Add Python to PATH" kutucuğunu işaretlemeyi unutmayın, yoksa hiçbir şey çalışmaz.

# Linux için kurulum
mkdir ile bir klasör oluşturun. Ondan sonra klasörde şu komutları sırayla girin:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run Proje.py

Tarayıcınızda açılan localhost ile projemizi kullanabilirsiniz.

# Windows için kurulum
Klasörde şu komutları sırayla girin:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run Proje.py

Tarayıcınızda açılan localhost ile projemizi kullanabilirsiniz.
