file = open("/home/mario/PycharmProjects/PredictionNSM/Results/penalty", "a+")
print(file.mode)
print(file.name)
file.write("www.youtube.com\n")
a = "www.facebook.com\n"
b = [1,1,1,1,1,1,1,1,1,1,1,1,1]
file.write(a)
file.write(str(b) + "\n")
file.write(a)
file.close()