import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    x = torch.randint(-10, 10, (34, ))

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(x)
    ax.set_xlabel('XLabel', fontproperties=Font().axis_labels, fontweight='bold')
    
    plt.show()


class Font():

    def __init__(self, family='serif', name='Times New Roman'):

        self.family = family
        self.name = name

        self.axis_labels = self.__init_axis_labels()

    def __init_axis_labels(self):
        font = FontProperties()
        font.set_family('serif')
        font.set_name('Times New Roman')
        font.set_style('italic')
        return font


if __name__ == "__main__":
    main()