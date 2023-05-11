while True:
    try:
        print("Unesite broj od 0.0 do 1.0")
        broj = float(input())
    except ValueError:
        print("Unos nije broj")
    else:
        if(broj>1.0 or broj<0.0):
            print("broj nije u rasponu!!")
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
