import torch
import rle_cuda

def encode(tensor, lens):
    countsOut, symbolsOut = rle_cuda.encode(tensor, lens)
    return countsOut, symbolsOut

def decode(countsOut, symbolsOut, grads):
    rle_cuda.decode(countsOut, symbolsOut, grads)

