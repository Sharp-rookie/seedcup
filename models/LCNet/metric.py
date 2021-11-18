from typing import List

import torch


def Precision(pred: torch.Tensor, y: torch.Tensor):
    """Precision calculation

    Args:
        pred (torch.Tensor): output label of model
        y (torch.Tensor): actual label 
    """

    # precision = TP / (TP + FP)

    index_ = pred == 1
    TP = (y[index_] == 1).sum()

    return (TP / index_.sum()).item()


def Recall(pred: torch.Tensor, y: torch.Tensor):
    """Recall calculation

    Args:
        pred (torch.Tensor): output label of model
        y (torch.Tensor): actual label 
    """

    # recall = TP / (TP + FN)

    index_ = y == 1
    TP = (pred[index_] == 1).sum()

    return (TP / index_.sum()).item()


def Accuracy(pred: torch.Tensor, y: torch.Tensor):
    """Accuracy calculation

    Args:
        pred (torch.Tensor): output label of model
        y (torch.Tensor): actual label 
    """

    return ((pred == y).sum() / len(y)).item()


def Fscore(pred: torch.Tensor, y: torch.Tensor):
    """Accuracy calculation

    Args:
        pred (torch.Tensor): output label of model
        y (torch.Tensor): actual label 
    """

    # F = 5PR / (2P + 3R)
    P = Precision(pred, y)
    R = Recall(pred, y)

    F = 5 * P * R / (2 * P + 3 * R)

    return F


def P1(pred: torch.Tensor, y: torch.Tensor):

    index_ = pred == 1
    TP = (y[index_] == 1).sum()

    return (TP / index_.sum()).item()


def P0(pred: torch.Tensor, y: torch.Tensor):

    index_ = pred == 0
    TP = (y[index_] == 0).sum()

    return (TP / index_.sum()).item()


def precision_list(pre_result: List[int], y_result: List[int]) -> float:
    TP = sum([1 if pre_result[i] == 1 and y_result[i] ==
             1 else 0 for i in range(len(pre_result))])
    FP = sum([1 if pre_result[i] == 1 and y_result[i] ==
             0 else 0 for i in range(len(pre_result))])
    return TP / (TP + FP)


def precision_list_(pre_result: List[int], y_result: List[int]) -> float:
    TP = sum([1 if pre_result[i] == 0 and y_result[i] ==
             0 else 0 for i in range(len(pre_result))])
    FP = sum([1 if pre_result[i] == 0 and y_result[i] ==
             1 else 0 for i in range(len(pre_result))])
    return TP / (TP + FP)


def recall_list(pre_result: List[int], y_result: List[int]) -> float:
    TP = sum([1 if pre_result[i] == 1 and y_result[i] ==
             1 else 0 for i in range(len(pre_result))])
    FN = sum([1 if pre_result[i] == 0 and y_result[i] ==
             1 else 0 for i in range(len(pre_result))])
    return TP / (TP + FN)


def f_score_list(pre_result: List[int], y_result: List[int]) -> float:
    P = precision_list(pre_result, y_result)
    R = recall_list(pre_result, y_result)
    return 5 * P * R / (2 * P + 3 * R)


if __name__ == "__main__":
    pred = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1])
    y = torch.cat([torch.zeros(8), torch.ones(8)])
    print(Precision(pred, y))
    print(Recall(pred, y))
    print(Accuracy(pred, y))
    print(Fscore(pred, y))
    print(Precision(y, y))
    print(Recall(y, y))
    print(Accuracy(y, y))
    print(Fscore(y, y))
