import matplotlib.pyplot as plt
import json


def visualization_acc(cell_type, train_acc1, test_acc1, train_auroc1, test_auroc1, train_auprc1, test_auprc1):
    plt.clf()
    x1 = [i * 10 for i in range(0, len(train_acc1), 1)]
    plt.plot(x1, train_acc1, label="train_acc")
    plt.plot(x1, test_acc1, label="test_acc")
    plt.plot(x1, train_auroc1, label="train_auroc")
    plt.plot(x1, test_auroc1, label="test_auroc")
    plt.plot(x1, train_auprc1, label="train_auprc")
    plt.plot(x1, test_auprc1, label="test_auprc")
    plt.legend(loc='lower right')
    plt.title(f"{cell_type} TF-region prediction result")
    plt.xlabel("epochs")
    plt.ylabel("metric")
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(f'./plot/{cell_type}_acc')


def visualization_loss(cell_type, train_loss1, test_loss1):
    plt.clf()
    x1 = [i * 10 for i in range(0, len(train_loss1), 1)]
    plt.plot(x1, train_loss1, label="train_loss")
    plt.plot(x1, test_loss1, label="test_loss")
    plt.legend(loc='lower right')
    plt.title(f"{cell_type} TF-region prediction loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    # plt.show()
    plt.savefig(f'./plot/{cell_type}_loss')


def read_result(path):
    with open(path, 'r') as file:
        data = json.load(file)

    train_loss = []
    test_loss = []

    train_acc = []
    train_auroc = []
    train_auprc = []
    test_acc = []
    test_auroc = []
    test_auprc = []

    for k in data.keys():
        train_loss.append(data[k]['train_loss'])
        test_loss.append(data[k]['test_loss_list'])

        train_acc1 = data[k]['train_acc_list']
        train_auroc1 = data[k]['train_auroc_list']
        train_auprc1 = data[k]['train_auprc_list']
        test_acc1 = data[k]['test_acc_list']
        test_auroc1 = data[k]['test_auroc_list']
        test_auprc1 = data[k]['test_auprc_list']

        train_acc1.insert(0,0)
        train_auroc1.insert(0,0)
        train_auprc1.insert(0,0)
        test_acc1.insert(0,0)
        test_auroc1.insert(0,0)
        test_auprc1.insert(0,0)

        train_acc.append(train_acc1)
        train_auroc.append(train_auroc1)
        train_auprc.append(train_auprc1)
        test_acc.append(test_acc1)
        test_auroc.append(test_auroc1)
        test_auprc.append(test_auprc1)

    return list(data.keys()), train_loss, train_acc, train_auroc, train_auprc, test_loss, test_acc, test_auroc, test_auprc


if __name__ == "__main__":
    path = './numerical_result/result.json'
    cell_types, train_loss, train_acc, train_auroc, train_auprc, test_loss, test_acc, test_auroc, test_auprc = read_result(path)
    for i in range(len(train_acc)):
        visualization_acc(cell_types[i], train_acc[i], train_auroc[i], train_auprc[i], test_acc[i], test_auroc[i], test_auprc[i])
        visualization_loss(cell_types[i], train_loss[i], test_loss[i])