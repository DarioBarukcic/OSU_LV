fhand = open("song.txt")
rijeci = {}

for line in fhand:
    line = line.rstrip()
    words = line.split()

    for word in words:
        if(word in rijeci):
            rijeci[word] += 1
        else:
            rijeci[word] = 1

fhand.close()

counter = 0

for word in rijeci:
    if(rijeci[word]==1):
        print(word)
        counter += 1

print(counter)