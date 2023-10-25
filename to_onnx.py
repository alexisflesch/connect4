import torch

from neural_networks import my_DDQN


def main():
    pytorch_model = my_DDQN()
    pytorch_model.load_state_dict(torch.load('models/model_123.pth'))
    pytorch_model.eval()
    dummy_input = torch.zeros(42)
    torch.onnx.export(pytorch_model,
                      dummy_input,
                      '123.onnx',
                      input_names=['input'],
                      output_names=['q'],
                      verbose=True)


if __name__ == '__main__':
    main()
