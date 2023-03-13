brojevi = []

while True:
    try:
        broj = input()
        if(broj=="Done"):
            break
        broj = float(broj)
    except ValueError:
        print("Unos nije broj ili Done")
    else:
        brojevi.append(broj)

if(len(brojevi)==0):
    print("Nije unesena ni jedan broj !!!")
else:
    print(f"Ukupno brojeva: {len(brojevi)}")
    print(f"Srednja vrijednost: {sum(brojevi)/len(brojevi)}")
    print(f"Minimalna vrijednost: {min(brojevi)}")
    print(f"Maksimalna vrijednost: {max(brojevi)}")

    brojevi.sort()

    print(brojevi)