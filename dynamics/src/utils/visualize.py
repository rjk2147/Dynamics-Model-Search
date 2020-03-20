import matplotlib.pyplot as plt
from pandas import DataFrame
import math
from IPython import display

def plot_velocity_curve(true, pred):
    data = DataFrame()
    data["y"] = true
    data["y_hat"] = pred
    data.plot(legend=True)
    plt.show()

def get_velocity_curve(title, true, pred):
    fig, ax = plt.subplots()
    ax.plot(true, c="g", label="y")
    ax.plot(pred, c="r", label="y_hat")
    ax.set_ylabel("velocity")
    ax.set_xlabel("timestep")
    return ax
    
def plot_velocity_curve_with_uncertainty(true, pred, error1, error2):
    data = DataFrame()
    data["y"] = true
    data["y_hat_mean"] = pred
    data["-std"] = error1
    data["+std"] = error2
    data.plot(legend=True)
    plt.show()

def plot_velocity_curve_with_uncertainty_shaded(true, pred, error1, error2):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True)
    ax.plot(pred)
    ax.fill_between(x=[i for i in range(pred.shape[0])], y1=error1, y2=error2, alpha=0.5)
    ax.plot(true)

def compute_link(angle1, angle2, quad, r=1):
    x1, y1 = (r*math.cos(angle1), r*math.sin(angle1))
    x2, y2 = (r*math.cos(angle1) + r*math.cos(angle1+angle2), r*math.sin(angle2) + r*math.sin(angle1+angle2))
    if quad == 2:
        x1 *= -1
        x2 *= -1
    elif quad == 3:
        x1 *= -1
        y1 *= -1
        x2 *= -1
        y2 *= -1
    elif quad == 4:
        y1 *= -1
        y2 *= -1
    return (x1, y1), (x2, y2)

def plot_links_side_by_side(true, pred):
    link1 = compute_link(true[0], true[1], quad=1)
    link2 = compute_link(true[2], true[3], quad=2)
    link3 = compute_link(true[4], true[5], quad=3)
    link4 = compute_link(true[6], true[7], quad=4)
    plt.subplot(1,2,1)
    plt.plot([0, link1[0][0], link1[1][0]], [0, link1[1][0], link1[1][1]])
    plt.plot([0, link2[0][0], link2[1][0]], [0, link2[1][0], link2[1][1]])
    plt.plot([0, link3[0][0], link3[1][0]], [0, link3[1][0], link3[1][1]])
    plt.plot([0, link4[0][0], link4[1][0]], [0, link4[1][0], link4[1][1]])
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout(pad=1)
    
    link1 = compute_link(pred[0], pred[1], quad=1)
    link2 = compute_link(pred[2], pred[3], quad=2)
    link3 = compute_link(pred[4], pred[5], quad=3)
    link4 = compute_link(pred[6], pred[7], quad=4)
    plt.subplot(1,2,2)
    plt.plot([0, link1[0][0], link1[1][0]], [0, link1[1][0], link1[1][1]])
    plt.plot([0, link2[0][0], link2[1][0]], [0, link2[1][0], link2[1][1]])
    plt.plot([0, link3[0][0], link3[1][0]], [0, link3[1][0], link3[1][1]])
    plt.plot([0, link4[0][0], link4[1][0]], [0, link4[1][0], link4[1][1]])
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout(pad=1)
    
    display.clear_output(wait=True)
    display.display(plt.gcf()) 
    plt.clf()

def plot_links_overlap(true, pred):
    link1 = compute_link(true[0], true[1], quad=1)
    link2 = compute_link(true[2], true[3], quad=2)
    link3 = compute_link(true[4], true[5], quad=3)
    link4 = compute_link(true[6], true[7], quad=4)
    plt.plot([0, link1[0][0], link1[1][0]], [0, link1[1][0], link1[1][1]], color="green")
    plt.plot([0, link2[0][0], link2[1][0]], [0, link2[1][0], link2[1][1]], color="green")
    plt.plot([0, link3[0][0], link3[1][0]], [0, link3[1][0], link3[1][1]], color="green")
    plt.plot([0, link4[0][0], link4[1][0]], [0, link4[1][0], link4[1][1]], color="green")
    
    link1 = compute_link(pred[0], pred[1], quad=1)
    link2 = compute_link(pred[2], pred[3], quad=2)
    link3 = compute_link(pred[4], pred[5], quad=3)
    link4 = compute_link(pred[6], pred[7], quad=4)
    plt.plot([0, link1[0][0], link1[1][0]], [0, link1[1][0], link1[1][1]], color="red")
    plt.plot([0, link2[0][0], link2[1][0]], [0, link2[1][0], link2[1][1]], color="red")
    plt.plot([0, link3[0][0], link3[1][0]], [0, link3[1][0], link3[1][1]], color="red")
    plt.plot([0, link4[0][0], link4[1][0]], [0, link4[1][0], link4[1][1]], color="red")
    
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout(pad=1)
    display.clear_output(wait=True)
    display.display(plt.gcf()) 
    plt.clf()

def compare_states(Y, Y_hat, overlap=False):
    for i in range(Y.shape[0]):
        if overlap:
            plot_links_overlap(true=Y[i, 5::2], pred=Y_hat[i, 5::2])
        else:
            plot_links_side_by_side(true=Y[i, 5::2], pred=Y_hat[i, 5::2])
