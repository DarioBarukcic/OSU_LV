print("Unesite ocjenu izmedju 0..0 i 1.0")

while True:
    try:
        broj = float(input())
    except ValueError:
        print("Unos nije broj")
    else:
        if(broj>1.0 or broj<0.0):
            print("Unos nije u zadanom intervalu")
            continue
        elif(broj>=0.9):
            print("A")
        elif(broj>=0.8):
            print("B")
        elif(broj>=0.7):
            print("C")
        elif(broj>=0.6):
            print("D")
        else:
            print("F")
        break
        