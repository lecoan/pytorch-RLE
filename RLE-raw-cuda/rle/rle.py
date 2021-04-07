import torch

import rle_cuda

torch.manual_seed(42)


class RLEFunction(object):
    @staticmethod
    def encode(input, input_int):
        countsOut, symbolsOut = rle_cuda.encode(input, input_int)
        return torch.squeeze(countsOut), torch.squeeze(symbolsOut)

    @staticmethod
    def decode(input):
        countsOut = input[0]
        symbolsOut = input[1]
        line = countsOut.size()[1] - 1
        result = torch.zeros((1, countsOut[0][line]), dtype=torch.float32, device=symbolsOut.get_device())
        return rle_cuda.decode(countsOut, symbolsOut, result)


class RLE(object):
    @staticmethod
    def encode(tensor, mask):
        tensor = tensor.view(1, -1)
        mask = mask.view(1, -1).int()
        counts, symbols = RLEFunction.encode(tensor, mask)
        print(symbols)
        symbols = torch.cat((symbols, torch.ones(1)))
        symbols = symbols.type_as(counts)
        return torch.stack((counts, symbols))

    @staticmethod
    def decode(tensor, size):
        counts = tensor[0]
        symbols = tensor[1].narrow(0, 0, counts.size().numel() - 1)
        symbols = symbols.type(torch.int64)
        output = RLEFunction.decode((counts, symbols))
        return output.reshape(size)

