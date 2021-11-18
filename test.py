import os
from functools import wraps

from torch.utils.data import DataLoader

from config.test_config import test_config
from LCNet_model import CTNet
from dataset.seed_dataset import SeedDataset
from models.baseline.baseline_model import Fake1DAttention
from res_model import *


def Model_Test():
    LC_weight, LC_outputs = LCNet_test()
    resnet_weight, resnet_outputs = ResNet_test()
    baseline_weight, baseline_outputs = baseline_test()
    total_weight = LC_weight + resnet_weight + baseline_weight
    # total_weight = LC_weight + resnet_weight
    result = [(LC_outputs[i] + resnet_outputs[i] + baseline_outputs[i]) / total_weight  for i in range(len(LC_outputs))]
    result = list(map(lambda x: '1' if x >= 0.5 else '0', result))

    with open("./history/weighted/weighted_result.txt", 'w') as f:
        f.write("\n".join(result))


def Test(config, name):
    def Test_Decorator(test_func):
        @wraps(test_func)
        def model_test():
            if name == "LCNet":
                model = CTNet(batch=1, in_channels=config.input_features, out_channels=config.output_features)
            elif name == "ResNet":
                model = ResNet(ResidualBlock, [2, 2, 2], config.input_features)
            else:
                model = Fake1DAttention(in_features=config.input_features, out_features=config.output_features)
            model.load_state_dict(torch.load(config.model, map_location=torch.device("cpu")))
            model.eval()

            test_dataset = SeedDataset(config.test_file)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            outputs = []
            for x in test_dataloader:
                logit = model(x)
                outputs.append(str(logit.argmax(1).item()))

            output_file_name = config.model.split("/")[-1].split(".")[0] + "_result.txt"
            output_file_path = name
            with open(os.path.join("./history/", output_file_path, output_file_name), 'w') as f:
                f.write('\n'.join(outputs))
            weight = eval(config.model.split("_")[-1].split(".")[-2]) / 10000
            return test_func(weight, outputs)

        return model_test

    return Test_Decorator


@Test(config=test_config.LCNet_config, name="LCNet")
def LCNet_test(weight, outputs):
    outputs = list(map(lambda x: eval(x) * weight, outputs))
    return weight, outputs


@Test(config=test_config.ResNet_config, name="ResNet")
def ResNet_test(weight, outputs):
    outputs = list(map(lambda x: eval(x) * weight, outputs))
    return weight, outputs


@Test(config=test_config.baseline_config, name="Fake1DAttention")
def baseline_test(weight, outputs):
    outputs = list(map(lambda x: eval(x) * weight, outputs))
    return weight, outputs


if __name__ == "__main__":
    Model_Test()
