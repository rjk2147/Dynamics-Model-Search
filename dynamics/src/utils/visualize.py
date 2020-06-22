import matplotlib.pyplot as plt
from pandas import DataFrame
import math
from IPython import display
import plotly.graph_objects as go


def plot_multimodel_velocity_curve(true, pred, axis=None):
    data = DataFrame()
    data["y"] = true
    for i in range(len(pred)):
        data["y_hat"+str(i)] = pred[i]
    data.plot(legend=True)
    if axis == 'equal':
        plt.axis('equal')
    plt.show()

def plot_multimodel_position_curve(true, pred, axis=None):
    data = DataFrame()
    data["y"] = true
    for i in range(len(pred)):
        data["y_hat"+str(i)] = pred[i]
    data.plot(legend=True)
    if axis == 'equal':
        plt.axis('equal')
    plt.show()


def plot_velocity_curve(true, pred, axis=None):
    data = DataFrame()
    data["y"] = true
    data["y_hat"] = pred
    data.plot(legend=True)
    if axis == 'equal':
        plt.axis('equal')
    plt.show()

def get_velocity_curve(title, true, pred):
    fig, ax = plt.subplots()
    ax.plot(true, c="g", label="y")
    ax.plot(pred, c="r", label="y_hat")
    ax.set_ylabel("velocity")
    ax.set_xlabel("timestep")
    return ax

def plot_position_curve(true, pred, axis=None):
    data = DataFrame()
    data["y"] = true
    data["y_hat"] = pred
    data.plot(legend=True)
    if axis == 'equal':
        plt.axis('equal')
    plt.show()

def plot_2D_position_curve(trueX, trueY, predX, predY, axis=None):
    fig, ax = plt.subplots()
    ax.plot(trueX, trueY, c="g", label="y")
    ax.plot(predX, predY, c="r", label="y")
    ax.set_ylabel("y-position")
    ax.set_xlabel("x-position")
  
    if axis == 'equal':
        plt.axis('equal')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.5)

    plt.show()

def plot_multimodel_2D_position_curve(trueX, trueY, predX, predY, xlabel='x', ylabel='y', axis=None):
    colors= ["r", "b", "y"]
    fig, ax = plt.subplots()
    ax.plot(trueX, trueY, c="g", label="y")
    for i in range(len(predX)):
        ax.plot(predX[i], predY[i], c=colors[i], label="yhat"+str(i))
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)

    if axis == 'equal':
        plt.axis('equal')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.5)

def get_position_curve(title, true, pred):
    fig, ax = plt.subplots()
    ax.plot(true, c="g", label="y")
    ax.plot(pred, c="r", label="y_hat")
    ax.set_ylabel("position")
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

def plot_curve_plotly(true, pred):
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=true,
                        mode='lines',
                        name='y'))
    fig.add_trace(go.Scatter(y=pred,
                        mode='lines',
                        name='y_hat'))
    fig.show()

def plot_curve_with_uncertainty_shaded_plotly(true, pred, error1, error2):
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pred,
                    mode='lines',
                    name='y_hat'))    

    fig.add_trace(go.Scatter(y=error2,
                        mode='lines',
                        name='upper bound'
                        ))

    fig.add_trace(go.Scatter(y=error1,
                        mode='lines',
                        name='lower bound'))
    
    fig.add_trace(go.Scatter(y=true,
                        mode='lines',
                        name='y'))
    
    fig.show()

def plot_xy_curve_with_uncertainty_shaded_plotly(true, pred, error1, error2):
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred[0], y=pred[1],
                    mode='lines',
                    name='y_hat'))    

    fig.add_trace(go.Scatter(x=error2[0], y=error2[1],
                        mode='lines',
                        name='upper bound'
                        ))

    fig.add_trace(go.Scatter(x=error1[0], y=error1[1],
                        mode='lines',
                        name='lower bound'))
    
    fig.add_trace(go.Scatter(x=true[0], y=true[1],
                        mode='lines',
                        name='y'))
    
    fig.show()

def plot_velocity_curve_with_uncertainty_shaded(true, pred, error1, error2, axis=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True)
    ax.plot(true)
    ax.plot(pred)
    ax.fill_between(x=[i for i in range(pred.shape[0])], y1=error1, y2=error2, alpha=0.5)
    if axis == 'equal':
        plt.axis('equal')

def plot_multimodel_velocity_curve_with_uncertainty_shaded(true, pred, error1, error2, axis=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True)
    ax.plot(true, label="y")
    for i in range(len(pred)):
        ax.plot(pred[i], label="yhat"+str(i))
        ax.fill_between(x=[i for i in range(true.shape[0])], y1=error1[i], y2=error2[i], alpha=0.5)

    if axis == 'equal':
        plt.axis('equal')

def plot_position_curve_with_uncertainty_shaded(true, pred, error1, error2, axis=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True)
    ax.plot(true, label="y")
    ax.plot(pred, label="yhat")
    ax.fill_between(x=[i for i in range(pred.shape[0])], y1=error1, y2=error2, alpha=0.5)
    if axis == 'equal':
        plt.axis('equal')

def plot_multimodel_position_curve_with_uncertainty_shaded(true, pred, error1, error2, axis=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True)
    ax.plot(true, label="y")
    for i in range(len(pred)):
        ax.plot(pred[i], label="yhat"+str(i))
        ax.fill_between(x=[i for i in range(true.shape[0])], y1=error1[i], y2=error2[i], alpha=0.5)
    if axis == 'equal':
        plt.axis('equal')


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

def plot_multimodel_links_side_by_side(true, pred):
    link1 = compute_link(true[0], true[1], quad=1)
    link2 = compute_link(true[2], true[3], quad=2)
    link3 = compute_link(true[4], true[5], quad=3)
    link4 = compute_link(true[6], true[7], quad=4)
    plt.subplot(1,1+len(pred),1)
    plt.title("y")
    plt.plot([0, link1[0][0], link1[1][0]], [0, link1[1][0], link1[1][1]])
    plt.plot([0, link2[0][0], link2[1][0]], [0, link2[1][0], link2[1][1]])
    plt.plot([0, link3[0][0], link3[1][0]], [0, link3[1][0], link3[1][1]])
    plt.plot([0, link4[0][0], link4[1][0]], [0, link4[1][0], link4[1][1]])
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout(pad=1)

    for i in range(len(pred)):
        link1 = compute_link(pred[i][0], pred[i][1], quad=1)
        link2 = compute_link(pred[i][2], pred[i][3], quad=2)
        link3 = compute_link(pred[i][4], pred[i][5], quad=3)
        link4 = compute_link(pred[i][6], pred[i][7], quad=4)
        plt.subplot(1,1+len(pred),2+i)
        plt.title("yhat"+str(i))
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

def compare_multimodel_states(Y, Y_hats):
    for i in range(Y.shape[0]):
        pred = [Y_hats[j][i, 5::2] for j in range(len(Y_hats))]
        plot_multimodel_links_side_by_side(true=Y[i, 5::2], pred=pred)

