import matplotlib.pyplot as plt
import csv
import os
from matplotlib.ticker import FormatStrFormatter



def plot():
    x_label = []
    iterations = []
    accuracies_top1 = []
    accuracies_top5 = []
    path = "./ckpt/test2/"

    for csv_file in os.listdir(path):
        if csv_file.endswith(".csv") and csv_file.find("metrics"):
            x_label.append(csv_file.split(".csv")[0].split("_")[-1])

            with open(path +"/"+ csv_file, newline="") as f:
                reader = list(csv.DictReader(f))

                iterations.append([int(row["iteration"]) for row in reader])
                accuracies_top1.append([float(row["top1"]) for row in reader])
                accuracies_top5.append([float(row["top5"]) for row in reader])
                # ZSCL_losses = [row["ZSCL"] for row in reader]
    
    # print(x_label)
    # print(iterations)
    # print(accuracies_top1)
    # print(accuracies_top5)
    # print(ZSCL_losses)

    index = 0
    for dataset in x_label:
        plt.plot(iterations[index], accuracies_top1[index], label=dataset)
        
        index+=1

    # plt.ylim(0,100)
    plt.title("accuracy over iterations\n \
            Trained on: DTD",
            fontsize=12)
    plt.legend()
    plt.ylabel("accuracy(%)")
    plt.xlabel("iterations")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    plt.savefig(path +"/"+ "output.png")

    # plt.plot(iterations, )


if __name__ == "__main__":
    plot()