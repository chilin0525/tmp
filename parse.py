import json
import sys

def nsys_parsing(files, result_path):
    for file_name in files:
        with open(file_name, newline='') as jsonfile:
            data = json.load(jsonfile)
        tmp = sys.stdout
        sys.stdout = open(result_path, "a+")

        total_time = 0.0
        gpu = []
        max_HtoD = 0
        max_DtoD = 0
        max_DtoH = 0
        other = 0.0

        for i in data:
            total_time += i["Duration(nsec)"]
            if i["Name"] == "[CUDA memcpy HtoD]":
                max_HtoD += i["Duration(nsec)"]
            elif i["Name"] == "[CUDA memcpy DtoD]":
                max_DtoD += i["Duration(nsec)"]
            elif i["Name"] == "[CUDA memcpy DtoH]":
                max_DtoH += i["Duration(nsec)"]
            else:
                other += i["Duration(nsec)"]
            if i["Device"] not in gpu:
                gpu.append(i["Device"])

        # print("|%s|%s|%s|%s|%s|%s|%s|%s|" % ("model name", "model time","gpu type", "HtoD","execution time", "DtoD", "DtoH", "runtime"))
        # print("|:---:"*8+"|")
        print("%s %s %s %f %f %f" % (file_name.split("-")[0], str(total_time/1000000.0),
                                                            gpu, max_HtoD/1000000.0, max_DtoD/1000000.0, max_DtoH/1000000.0))
        sys.stdout = tmp

    sys.stdout.close()
    sys.stdout = tmp


def main():

    gtx1080 = ["bert-GTX1080_gputrace.json",
               "googlenet-GTX_1080_Ti_gputrace.json", "resnet50-GTX_1080_Ti_gputrace.json"]
    rtx2060 = ["bert-RTX2060_gputrace.json",
               "googlenet-RTX_2060_gputrace.json", "resnet50-RTX_2060_gputrace.json", "ssd-RTX2060_gputrace.json"]

    nsys_parsing(gtx1080, "gtx1080")
    nsys_parsing(rtx2060, "rtx2060")

if __name__ == "__main__":
    main()
