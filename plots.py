from matplotlib import pyplot as plt
import json

if __name__ == '__main__':
    history = json.load(open('lr_finder.json'))
    lrs = history['lr'][40:-4]
    losses = history['loss'][40:-4]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    plt.show()