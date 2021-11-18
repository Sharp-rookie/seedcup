import argparse


class ForestDecisionConfig:
    def __init__(self, configs):
        self.criterion = configs.criterion
        self.splitter = configs.splitter
        self.random_state = configs.random_state
        self.max_depth = configs.max_depth
        self.test = configs.test
        self.model = configs.model
        self.test_data = configs.test_data
        self.train_data = configs.train_data
        self.valid_data = configs.valid_data
        # self.min_samples_split = 2,
        # self.min_samples_leaf = 1,
        # self.min_weight_fraction_leaf = 0.,
        # self.max_features = None,
        # self.random_state = None,
        # self.max_leaf_nodes = None,
        # self.min_impurity_decrease = 0.,
        # self.min_impurity_split = None,
        # self.class_weight = None,
        # self.presort = 'deprecated',
        # self.ccp_alpha = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", help="max-depth of the decision tree", type=int, default=5)
    parser.add_argument("--random-state", help="random state of the decision tree", type=int, default=0)
    parser.add_argument("--criterion", "-c", help="judge criterion of the tree", type=str, choices=['gini', 'entropy'],
                        default='gini')
    parser.add_argument("--splitter", "-s", help="splitter strategy of the tree", type=str, choices=['best', 'random'],
                        default='best')
    parser.add_argument("--test", "-t", help="train or test", type=bool, default=False)
    parser.add_argument("--model", help="path to saved model", type=str, default='./history/ML/DecisionTree_max-depth=3_0.8142.pkl')
    parser.add_argument("--train-data", help="path to train data", type=str, default="./data/ML/33_dimension/train.csv")
    parser.add_argument("--test-data", help="path to test data", type=str, default="./data/ML/33_dimension/test_b.csv")
    parser.add_argument("--valid-data", help="path to valid data", type=str, default="./data/ML/33_dimension/valid.csv")
    return parser.parse_args()



arguments = parse_args()
config = ForestDecisionConfig(arguments)
