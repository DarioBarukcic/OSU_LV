spamCounters = []
hamCounters = []
usklicnikCounter = 0

fhand = open("SMSSpamCollection.txt",encoding="utf8")

for line in fhand:
    line = line.rstrip()
    words = line.split()

    if(words[0]=="spam"):
        spamCounters.append(len(words)-1)
        if(line.endswith("!")):
            usklicnikCounter +=1
    else:
        hamCounters.append(len(words)-1)

fhand.close()

print(f"Spam prosjek: {sum(spamCounters)/len(spamCounters)}")
print(f"Ham prosjek: {sum(hamCounters)/len(hamCounters)}")
print(f"Broj rijeci koje zavrsavaju usklicnikom: {usklicnikCounter}")