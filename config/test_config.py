import argparse


class Config(object):
    def __init__(self, model="", test_file="", input_features=0, output_features=0):
        self.model: str = model
        self.test_file: str = test_file
        self.input_features: int = input_features
        self.output_features: int = output_features


class TestConfig(object):
    def __init__(self, **kwargs):
        self.LCNet_config = Config(model=configs.LC_model, test_file=configs.LC_input,
                                   input_features=configs.LC_input_f, output_features=configs.LC_output_f)
        self.ResNet_config = Config(model=configs.resnet_model, test_file=configs.resnet_input,
                                    input_features=configs.resnet_input_f, output_features=configs.resnet_output_f)
        self.baseline_config = Config(model=configs.baseline_model, test_file=configs.baseline_input,
                                      input_features=configs.baseline_input_f,
                                      output_features=configs.baseline_output_f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LC-input", help="input of LCNet", type=str, default="./data/33_dimension/test.csv")
    parser.add_argument("--baseline-input", help="input of baseline", type=str, default="./data/unmodified/test_a.csv")
    parser.add_argument("--resnet-input", help="input of resnet", type=str, default="./data/unmodified/test_a.csv")

    parser.add_argument("--LC-model", help="model of LCNet", type=str,
                        default="./history/LCNet/33维(未归一化)_131轮_0.8226.pt")
    parser.add_argument("--baseline-model", help="model of baseline", type=str, default="./checkpoints/unevol/29_epoc.pt")
    parser.add_argument("--resnet-model", help="model of resnet", type=str,
                        default="./history/ResNet/28维(未归一化)_273轮_0.8196.pt")

    parser.add_argument("--LC-input-f", help="input features of LCNet", type=int, default=33)
    parser.add_argument("--baseline-input-f", help="input features of baseline", type=int, default=28)
    parser.add_argument("--resnet-input-f", help="input features of resnet", type=int, default=28)

    parser.add_argument("--LC-output-f", help="output features of LCNet", type=int, default=2)
    parser.add_argument("--baseline-output-f", help="output features of baseline", type=int, default=2)
    parser.add_argument("--resnet-output-f", help="output features of resnet", type=int, default=2)

    return parser.parse_args()


configs = parse_args()
test_config = TestConfig()
