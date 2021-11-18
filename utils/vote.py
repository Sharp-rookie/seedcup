import os


def vote(directory="../history"):
    models = os.listdir(directory)
    models = list(filter(lambda x: x != 'weighted', models))
    print(models)
    # 365 is the number of the test sample
    scores = [0] * 365
    weights = []
    for model in models:
        # get the highest file of each model
        highest_result = sorted(os.listdir(os.path.join(directory, model)),
                                key=lambda x: x.split("_")[-1].split(".")[-2])
        highest_result = list(
            filter(lambda x: x.endswith(".txt"), highest_result))[-1]
        print(highest_result)
        # open the file
        with open(os.path.join(directory, model, highest_result), "r") as f:
            # get the weight from the file name
            weight = float(highest_result.split(
                "_")[-1].split(".")[-2]) / 10000
            # remember the weights of each model
            weights.append(weight)
            # add the score to the final result
            result = list(map(lambda x: int(x.replace('\n', ''))
                          * weight, f.readlines()))
            scores = [scores[i] + result[i] for i in range(len(scores))]
    # If the weighted score is greater than 0.5, it is 1, else it is 0
    scores = list(map(lambda x: '1' if x / sum(weights)
                  >= 0.5 else '0', scores))
    with open("../history/weighted/weighted_result.txt", "w") as f:
        f.write("\n".join(scores))


if __name__ == "__main__":
    vote()
