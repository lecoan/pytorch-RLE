import torch, rle


def main():
    a = rle.RLE()
    tensor1 = torch.tensor([[0.0, 0.2], [0.1, 0.0], [0.0, 0.0], [0.6, 0.0]]).cuda()
    tensor2 = torch.randn(307200, 1).cuda()
    b = [tensor1, tensor2]
    print(b)
    tensor_mask = [torch.tensor([[0, 1], [1, 0], [0, 0], [1, 0]]).cuda(), torch.ones_like(tensor2).cuda()]
    result = a.encode(b, tensor_mask)
    print(result)
    a.sizes = [(torch.Size([1, tup[0][0][tup[0].size()[1] - 1]])) for tup in result]
    result = a.decode(result, a.sizes)
    result_size = [(tensor.size()) for tensor in result]
    print("decode result={},size={}".format(result, result_size))

    a = rle.RLE()
    tensor1 = torch.tensor([[0.0, 0.2], [0.1, 0.0], [0.0, 0.0], [0.6, 0.0]]).cuda()
    tensor2 = torch.randn(307200, 1).cuda()
    b = [tensor1, tensor2]
    print(b)
    tensor_mask = [torch.tensor([[0, 1], [1, 0], [0, 0], [1, 0]]).cuda(), torch.ones_like(tensor2).cuda()]
    result = a.encode(b, tensor_mask)
    print(result)
    a.sizes = [(torch.Size([1, tup[0][0][tup[0].size()[1] - 1]])) for tup in result]
    result = a.decode(result, a.sizes)
    result_size = [(tensor.size()) for tensor in result]
    print("decode result={},size={}".format(result, result_size))


def main2():
    a = rle.RLE()
    tensor1 = torch.tensor([[0.0, 0.2], [0.1, 0.0], [0.0, 0.0], [0.6, 0.0]]).cuda()
    tensor2 = torch.randn(307200, 1).cuda()
    b = [tensor1, tensor2]
    print(b)
    tensor_mask = [torch.tensor([[0, 1], [1, 0], [0, 0], [1, 0]]).cuda(), torch.ones_like(tensor2).cuda()]
    result = a.encode(b, tensor_mask)
    print(result)


if __name__ == '__main__':
    main()
