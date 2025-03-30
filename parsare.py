dict = {}
with open("databaseBIG.txt",'r',encoding="utf-8") as file:
    for line in file.readlines():
        linesplit=line.split(":")
        key=linesplit[0][1:-1]
        value = linesplit[1][2:-2].split(',')
        value1 = value[0]
        value2 = value[1][1:-1]
        value = int(value1) + int(value2)
        if (value > 1000):
            if (int(value1)/value > 0.95 or int(value1)/value < 0.05):
                dict[key] = int(value1)/value

# for item in dict:
#     print(item,dict[item])
print(len(dict))

with open("aparitii.txt", "w") as file:
    for item in dict:
        file.write(item + "\n")

# with open("Hackathon\\fisier.txt",'r',encoding="utf-8") as file:
#     for line in file.readlines():
#         linesplit=line.split(":")
#         key=linesplit[0][1:-1]
#         value = linesplit[1][1:-2]
#         dict[key]=value

# for item in dict:
#      print(item,dict[item])