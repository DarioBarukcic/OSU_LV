
"""
print(int(input("Broj sati: "))*float(input("Cijena sata: ")))"""

def total_euro():
    return brojSati*cijenaSata

while True:
    try: 
        print("Broj sa3eti: ")
        brojSati = int(input())
        print("Cijena sata: ")
        cijenaSata = float(input())
    except ValueError:
        print("Unos nije broj!!!")
    else:
        print(total_euro())
        break







