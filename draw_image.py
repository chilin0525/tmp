import matplotlib.pyplot as plt
import numpy as np

def draw(x,y,name,x_name,y_name):
    plt.clf()
    x_idx = np.arange(len(x))
    plt.bar(x_idx, y)
    plt.xticks(x_idx,x)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # plt.ylim([y_min,y_max])
    plt.title(name)
    plt.savefig(name)

if __name__=="__main__":
    with open("gtx1080","r", newline="") as f:
        data = f.read().splitlines()
    modelname = list()
    h2d = list()
    d2d = list()
    d2h = list()
    total = list()
    for i in data:
        i = i.split(" ")
        modelname.append(i[0])
        total.append(float(i[1]))
        h2d.append(float(i[7]))
        d2d.append(float(i[8]))
        d2h.append(float(i[9]))
    draw(modelname, total, "GTX1080 total inference time", "model name", "time(ms)")
    draw(modelname, h2d, "GTX1080 total HtoD time", "model_name", "time(ms)")
    draw(modelname, d2d, "GTX1080 total DtoD time", "model_name", "time(ms)")
    draw(modelname, d2h, "GTX1080 total DtoH time", "model_name", "time(ms)")
    
    with open("rtx2060", "r", newline="") as f:
        data = f.read().splitlines()
    modelname = list()
    h2d = list()
    d2d = list()
    d2h = list()
    total = list()
    for i in data:
        i = i.split(" ")
        modelname.append(i[0])
        total.append(float(i[1]))
        h2d.append(float(i[6]))
        d2d.append(float(i[7]))
        d2h.append(float(i[8]))
    print(h2d)
    print(d2d)
    print(d2h)
    draw(modelname, total, "RTX2060 total inference time", "model name", "time(ms)")
    draw(modelname, h2d, "RTX2060 total HtoD time", "model_name", "time(ms)")
    draw(modelname, d2d, "RTX2060 total DtoD time", "model_name", "time(ms)")
    draw(modelname, d2h, "RTX2060 total DtoH time", "model_name", "time(ms)")
