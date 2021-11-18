from typing import Tuple

import sklearn.ensemble as ensemble
import numpy as np
import time
import pickle
from forest_decision_config import config
from metric import *
from dataset.ML_dataset import MLTestSeedDataset, MLTrainSeedDataset


def TrainModel() -> None:
    model = ensemble.RandomForestClassifier(max_depth=config.max_depth, random_state=config.random_state)
    dataset = MLTrainSeedDataset(config.train_data)
    model = model.fit(dataset.X, dataset.Y)
    ValidModel(model)
    current_time = str(time.strftime("%d-%H-%M-%S", time.localtime()))
    with open('treeCheckpoints\\forest' + current_time + '.pkl', 'wb') as output:
        pickle.dump(model, output)
        print("The model file is saved to " + 'treeCheckpoints/' +
              current_time + '.pkl')


def ValidModel(model, Dataset=None) -> Tuple[float, float]:
    if not Dataset:
        dataset = MLTrainSeedDataset(config.valid_data)
    else:
        dataset = MLTrainSeedDataset(Dataset)
    result = model.predict(dataset.X)
    total_num = len(result)
    acc = np.sum(np.array(result) - np.array(dataset.Y) == 0) / total_num
    P1, P0, R, F_score = precision_list(result, dataset.Y), precision_list_(result, dataset.Y), recall_list(
        result, dataset.Y), f_score_list(result, dataset.Y)
    print("The valid accuracy ============>", acc, "%")
    print("The valid P1 ============>", P1, "%")
    print("The valid P0 ============>", P0, "%")
    print("The valid recall ============>", R, "%")
    print("The valid f_score ============>", F_score, "%")
    # return acc, P1, R, F_score
    return P1, P0


def TestModel():
    with open(config.model, 'rb') as input:
        model = pickle.load(input)
    dataset = MLTestSeedDataset(config.test_data)
    result = model.predict(dataset.X)
    result = map(lambda x: str(x), result)
    with open('../../history/ML/33_forest_01-21-20-34_result.txt', "w") as f:
        f.write("\n".join(result))


if __name__ == "__main__":

    # clf, kernel, degree = args.clf, args.kernel, args.degree
    # criterions, max_depths, ccp_alphas = ['gini', 'entropy'], range(2, 10), range(0, 100)
    # result_file = pd.DataFrame(columns=['criterion', 'max-depth', 'ccp-alpha', 'Precision', 'Recall', 'Fscore'])
    #
    # i = 0
    # for max_depth in max_depths:
    #     for criterion in criterions:
    #         for ccp_alpha in ccp_alphas:
    #             ccp_alpha /= 100
    #             model = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
    #                                                 ccp_alpha=ccp_alpha)
    #             dataset = TreeTrainSeedDataset(config.train_data)
    #             model = model.fit(dataset.X, dataset.Y)
    #             acc, precision, recall, f_score = ValidModel(model)
    #             print(f"[criterion = {criterion} max-depth = {max_depth} ccp_alpha = {ccp_alpha}]\n Precision: {precision}\tRecall: {recall}\tFscore: {f_score}\n")
    #             result_file.loc[i] = (str(criterion), str(max_depth), str(ccp_alpha), str(precision), str(recall), str(f_score))
    #             i += 1
    #
    # result_file.to_csv("auto_result.csv")
    if config.test:
        TestModel()
    else:
        TrainModel()
