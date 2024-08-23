from dataset import *
from train import *
import torch
from genotypes import *
import network


def search(search_num):
    path = './config.yaml'
    config = read_yaml(path)
    device="cuda:0"
    best_model_dir = './results/trainmodel-2024-08-17-18-50-59/best_model_10.pth.tar'
    
    checkpoint = torch.load(best_model_dir)
    model = network.model()
    model.load_state_dict(checkpoint['model_state_dict'])

    print(torch.cuda.is_available())

    device = torch.device("cuda:0") # 或者 "cpu"
    model.to(device)

    # for param in model.parameters():
    #     if param.requires_grad:
    #         # 这里以添加均匀分布噪声为例，你可以调整噪声的大小
    #         noise = torch.randn(param.size()) * 0.01
    #         param.data += noise.to(device)
    
    dataset = DataSetDarts(0).generate_random_dataset(search_num)
    acc, tup = test_batch(model, dataset, config, device)
    print("Best acc: {:05f}, Best tuple: {}".format(acc, str(tup)))
    print(transfer_geno(tup))


def transfer_geno(best_tuple):
    normal_cell = best_tuple[0]
    normal_tup = []
    for (node, op) in normal_cell:
        normal_tup.append((PRIMITIVES[op], node))
    reduc_cell = best_tuple[1]
    reduc_tup = []
    for (node, op) in reduc_cell:
        reduc_tup.append((PRIMITIVES[op], node))
    cifar = Genotype(normal=normal_tup, normal_concat=[2, 3, 4, 5], reduce=reduc_tup, reduce_concat=[2, 3, 4, 5])
    return cifar


if __name__ == "__main__":
    search(100000)