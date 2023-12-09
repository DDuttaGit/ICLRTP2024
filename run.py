
def readfile():
    d = {}
    with open("config") as f:
    	lines = f.readlines()
    	for line in lines[:-1]:
    	    key, value = line[:-1].split(" ")
    	    d[key] = value
    
    return d

d = readfile()
if d["GAN"] == "VGAN":
    from VGAN import *
elif d["GAN"] == "WGAN":
    from WGAN import *
elif d["GAN"] == "LSGAN":
    from LSGAN import *

train(d["EPOCHS"], d["NOISE"], d["MU"], d["SIGMA"])
