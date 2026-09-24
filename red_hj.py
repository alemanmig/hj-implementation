import numpy as np
import matplotlib.pyplot as plt

def main():
    f = np.arange(0,30 * np.pi+0.01,0.01)
    x = (np.sin(f * 0.8)) / 1
    y = (np.sin(f * 1)) / 1

    fig, ax = plt.subplots()
    ax.plot(x)
    ax.plot(y)
    fig.savefig("red_hj_x.pdf", dpi=300)
    plt.close(fig)

 #   fig, ax = plt.subplots()
 #   ax.plot(y)
 #   fig.savefig("red_hj_y.pdf", dpi=300)
 #   plt.close(fig)

if __name__ == "__main__":
    main()