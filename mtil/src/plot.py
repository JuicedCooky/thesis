import matplotlib.pyplot as plt
import csv
import os

def plot():
    x_label = []
    iterations = []
    accuracies_top1 = []
    accuracies_top5 = []
    path = "./ckpt/test2/"

    for csv_file in os.listdir(path):
        if csv_file.endswith(".csv"):
            x_label.append(csv_file.split(".csv")[0])

            with open(path +"/"+ csv_file, newline="") as f:
                reader = list(csv.DictReader(f))

                iterations.append([row["iteration"] for row in reader])
                accuracies_top1.append([row["top1"] for row in reader])
                accuracies_top5.append([row["top5"] for row in reader])
                # ZSCL_losses = [row["ZSCL"] for row in reader]
    
    print(x_label)
    print(iterations)
    print(accuracies_top1)
    print(accuracies_top5)
    # print(ZSCL_losses)

    for dataset in x_label:
        index = 0
        plt.plot(iterations[index], accuracies_top1[index])
        
        plt.show()
        index+=1

    plt.plot(iterations, )


if __name__ == "__main__":
    plot()