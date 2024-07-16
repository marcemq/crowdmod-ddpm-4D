import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from data import filterDataByLU
ROWS=12
COLS=36
def drawMacroProps(crowd, info, maxRho, saveImg=False):
    """
    This function plots crowd macroproperties.
    """
    x, y = np.mgrid[0:crowd.cols, 0:crowd.rows]
    fig, ax = plt.subplots()
    ax.set_xlabel(f"frame:{info[0]} rho:{info[1]}")
    axp=ax.matshow(crowd.rho, cmap=plt.cm.Blues)
    Q = ax.quiver(crowd.mu_v[0,:,:], -crowd.mu_v[1,:,:], color='green', angles='xy',scale_units='xy', scale=1)
    cbar=plt.colorbar(axp, cmap=plt.cm.Blues, fraction=0.017, pad=0.04)
    cbar.mappable.set_clim(0, vmax=maxRho)

    for i in range(crowd.rows):
        for j in range(crowd.cols):
            center = (x[j,i]+crowd.mu_v[0,i,j], y[j,i]-crowd.mu_v[1,i,j])
            circle = plt.Circle(center, np.sqrt(crowd.sigma2_v[i,j]), fill=False, color='green')
            Q.axes.add_artist(circle)
    #plt.show()
    if saveImg:
        fig.savefig("images/macroProperties-"+str(info[0])+".svg", format='svg', bbox_inches='tight')
    return fig

def drawPredMacroProps(crowd_hat, crowd_gt, info, maxRho, drawUncGt, drawUncHat, saveImg=False):
    """
    This function plots crowd macroproperties.
    """
    x, y = np.mgrid[0:crowd_gt.cols, 0:crowd_gt.rows]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # plot GT and estimated macroprops
    ax1.set_xlabel(f"frame:{info[0]} rho_gt:{info[2]}")
    axp1 = ax1.matshow(crowd_gt.rho, cmap=plt.cm.Blues)
    Q1 = ax1.quiver(crowd_gt.mu_v[0,:,:], -crowd_gt.mu_v[1,:,:], color='green', angles='xy',scale_units='xy', scale=1)
    cbar1 = plt.colorbar(axp1, cmap=plt.cm.Blues, fraction=0.017, pad=0.04)
    cbar1.mappable.set_clim(0, vmax=maxRho)

    # plot estimated macroprops
    ax2.set_xlabel(f"frame:{info[0]} rho_hat:{info[1]}")
    axp2 = ax2.matshow(crowd_hat.rho, cmap=plt.cm.Blues)
    Q2 = ax2.quiver(crowd_hat.mu_v[0,:,:], -crowd_hat.mu_v[1,:,:], color='green', angles='xy',scale_units='xy', scale=1)
    cbar2 = plt.colorbar(axp2, cmap=plt.cm.Blues, fraction=0.017, pad=0.04)
    cbar2.mappable.set_clim(0, vmax=maxRho)

    for i in range(crowd_gt.rows):
        for j in range(crowd_gt.cols):
            center = (x[j,i]+crowd_gt.mu_v[0,i,j], y[j,i]+crowd_gt.mu_v[1,i,j])
            circle = plt.Circle(center, np.sqrt(crowd_gt.sigma2_v[i,j]), fill=False, color='green')
            if drawUncGt:
                Q1.axes.add_artist(circle)

            center = (x[j,i]+crowd_hat.mu_v[0,i,j], y[j,i]+crowd_hat.mu_v[1,i,j])
            circle = plt.Circle(center, np.sqrt(crowd_hat.sigma2_v[i,j]), fill=False, color='green')
            if drawUncHat:
                Q2.axes.add_artist(circle)

    if saveImg:
        fig.savefig("images/macroProperties-"+str(info[0])+".png")
    return fig

def plotPeopleDensity(x, y, LU, samplesToPlot, title, customScale=True, saveImg=True):
    """
    This function plots density given by x,y along with a grid that starts at LU point.
    And depending on the setting it arranges the x-axis and y-axis limits and ticks
    """
    x ,y = np.array(x), np.array(y)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    # Data sampling
    random_idx = np.random.choice(len(x), size=samplesToPlot, replace=False)
    x_rdm = x[random_idx]
    y_rdm = y[random_idx]
    ax.scatter(x_rdm, y_rdm, s=0.5)
    ax.set_title(title)
    # Grid creation and its plot
    gx, gy = np.meshgrid(np.linspace(LU[0], LU[0]+COLS, COLS + 1), np.linspace(LU[1], LU[1]-ROWS, ROWS + 1))
    ax.plot(gx, gy,c='green',linewidth=0.5)
    for i in range(ROWS + 1):
        ax.plot(gx[i,:], gy[i,:],c='green',linewidth=0.5)

    if customScale:
        x_ticks = range(-40, 60, 10)  # Generate x-axis tick positions every 10 units
        y_ticks = range(-30, 30, 10)
        ax.set_xticks(x_ticks), ax.set_yticks(y_ticks)
        x_limit = (-45, 60) 
        y_limit = (-30, 25)
        ax.set_xlim(x_limit), ax.set_ylim(y_limit)
        # plot LU
        ax.scatter(LU[0], LU[1], color='red', marker='o')
        ax.annotate(f'({"{:.4f}".format(LU[0])}, {"{:.4f}".format(LU[1])})', (LU[0], LU[1]), textcoords="offset points", xytext=(0,10), ha='center')
    else:
        ax.set_xticks([]), ax.set_yticks([])

    plt.show()
    if saveImg:
        fig.savefig("images/PeopleDensityWithGrid.png")

def plotPeopleDensityWithGridRotation(filename, LU=[12, -15], saveImg=True, theta=2.5647):
    fig, ax = plt.subplots(figsize=(12, 8))
    colNames =['time', 'personID', 'pos_x', 'pos_y', 'pos_z', 'vel', 'motion_angle', 'facing_angle']
    readColNames =['time', 'personID', 'pos_x', 'pos_y', 'vel', 'motion_angle']
    df = pd.read_csv(filename, names=colNames, header=None, usecols=readColNames)
    random_rows = df.sample(n=20000, random_state=42)
    ax.scatter(random_rows['pos_x']/1000, random_rows['pos_y']/1000, s=0.5)

    x, y = np.meshgrid(np.linspace(0, COLS, COLS + 1), np.linspace(0, ROWS, ROWS + 1))

    # Rotate the grid using the rotation matrix
    x_rot = x * np.cos(theta) - y * np.sin(theta) + LU[0]
    y_rot = x * np.sin(theta) + y * np.cos(theta) + LU[1]

    # plot vertical and horizontal grid lines
    ax.plot(x_rot, y_rot,c='green',linewidth=0.5)
    for i in range(ROWS + 1):
        ax.plot(x_rot[i,:], y_rot[i,:],c='green',linewidth=0.5)

    # plot LU
    plt.scatter(LU[0], LU[1], color='red', marker='o')
    plt.annotate(f'({LU[0]}, {LU[1]})', (LU[0], LU[1]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.show()
    if saveImg:
        fig.savefig("images/PeopleDensityWithGridRotation.png")

def plotDataAndItsRotation(ox, oy, rx, ry, oLU, rLU, theta, figName, saveImg=True):
    """
    This function have two subplot. At the top the original density with grid starting
    at oLU point, and at the bottom the rotated density with grid starting at LU.
    The params rx, ry, rLU are already rotated.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.scatter(ox, oy, s=0.5)
    ax1.set_title("Original Data")

    ax2.scatter(rx, ry,  s=0.5)
    ax2.set_title("Rotated Data by " + "{:.4f}".format(theta) + " radians")

    x_ticks = range(-40, 60, 10)  # Generate x-axis tick positions every 10 units
    y_ticks = range(-30, 30, 10)
    ax1.set_xticks(x_ticks), ax1.set_yticks(y_ticks)
    ax2.set_xticks(x_ticks), ax2.set_yticks(y_ticks)
    ax1.set_aspect('equal'), ax2.set_aspect('equal')

    x_limit = (-45, 60) 
    y_limit = (-30, 25)
    ax1.set_xlim(x_limit), ax1.set_ylim(y_limit)
    ax2.set_xlim(x_limit), ax2.set_ylim(y_limit)
 
    ogx, ogy = np.meshgrid(np.linspace(oLU[0], oLU[0]+COLS, COLS + 1), np.linspace(oLU[1], oLU[1]-ROWS, ROWS + 1))
    rgx, rgy = np.meshgrid(np.linspace(rLU[0], rLU[0]+COLS, COLS + 1), np.linspace(rLU[1], rLU[1]-ROWS, ROWS + 1))
    # plot grids
    ax1.plot(ogx, ogy,c='green',linewidth=0.5)
    ax2.plot(rgx, rgy,c='green',linewidth=0.5)
    for i in range(ROWS + 1):
        ax1.plot(ogx[i,:], ogy[i,:],c='green',linewidth=0.5)
        ax2.plot(rgx[i,:], rgy[i,:],c='green',linewidth=0.5)

    ax1.scatter(oLU[0], oLU[1], color='red', marker='o')
    ax1.annotate(f'({"{:.4f}".format(oLU[0])}, {"{:.4f}".format(oLU[1])})', (oLU[0], oLU[1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax2.scatter(rLU[0], rLU[1], color='red', marker='o')
    ax2.annotate(f'({"{:.4f}".format(rLU[0])}, {"{:.4f}".format(rLU[1])})', (rLU[0], rLU[1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.show()
    if saveImg:
        fig.savefig("images/" + figName +".png")

def plotDensityAndGrid(data, day, time_t, LU, saveImg=True):
    #AR: edit this plot to plot both in separate figures in order
    # to avoid filtering data in this function
    fig, ax = plt.subplots(nrows=2, ncols=1)
    # Plot pedestrian XY positions
    ax[0].scatter(data['pos_x'], data['pos_y'], s=1)

     # Add plot details
    ax[0].set_title(DAYS[day] + " at " + time_t.strftime('%Y-%m-%d %H:%M:%S'))
    ax[0].set_xlabel("pos_x"), ax[0].set_xticks(np.arange(-40, 51, 10.0))
    ax[0].set_ylabel("pos_y"), ax[0].set_yticks(np.arange(-25, 26, 5.0))
    fig.set_size_inches(8, 6)

    x, y = np.meshgrid(np.linspace(LU[0], LU[0] + COLS, COLS + 1), np.linspace(LU[1], LU[1] - ROWS, ROWS + 1))
    # plot vertical and horizontal grid lines
    ax[0].plot(x,y,c='green',linewidth=0.5)
    for i in range(ROWS + 1):
        ax[0].plot(x[i,:],y[i,:],c='green',linewidth=0.5)

    ########### plot data in grid: a zoom view of data
    ax[1].plot(x,y,c='green',linewidth=0.5)
    for i in range(ROWS + 1):
        ax[1].plot(x[i,:],y[i,:],c='green',linewidth=0.5)
    
    dataInGrid = filterDataByLU(data=data, LU=LU)
    ax[1].scatter(dataInGrid['pos_x'], dataInGrid['pos_y'], s=1)
    # plot vel_x and vel_y vectors
    ax[1].quiver(dataInGrid['pos_x'], dataInGrid['pos_y'], dataInGrid['vel_x'], dataInGrid['vel_y'], color='red', angles='xy',scale_units='xy', scale=1)

    plt.show()
    if saveImg:
        fig.savefig("images/DensityAndVelInGrid.png")

def plot_losses(list_loss_train, list_loss_val, subtitle, title="Overall"):
    plt.figure(figsize=(7,7))
    plt.plot(list_loss_train, label="train loss")
    plt.plot(list_loss_val, label="val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title+" Train/Val losses", fontsize=16)
    plt.suptitle(subtitle, fontsize=12)
    plt.legend()
    plt.savefig("images/loss_"+subtitle+".png")

if __name__ == '__main__':
    plotPeopleDensityWithGridRotation(filename='/Users/marcemq/Documents/PHD/inData/atc-20121114.csv', LU=[38.2789,-15.8076])