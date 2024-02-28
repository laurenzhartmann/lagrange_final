import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_bvp

def n(y):
    return 1.2 + 1 * y

# Define the differential equation
def ode(x, y):
    y0, y1 = y
    dydx = [y1, 1*(1 + y1**2)/ (1.2 + 1 * y0)]  # y0 = y, y1 = y'
    return dydx

# Define the boundary conditions
def bc(ya, yb):
    return np.array([ya[0]-0, yb[0] - 1])  # y(0) = 0, y(1) = 1

# Define the interval
x_bvp = np.linspace(0, 1, 100)
y_guess = (x_bvp, np.ones(x_bvp.size))
sol_bvp = solve_bvp(ode, bc, x_bvp, y_guess)

class DraggablePlotExample(object):
    u""" An example of plot with draggable markers """

    def __init__(self):
        self._figure, self._axes, self._line, self._scatter, self._foreground = None, None, None, None, None
        self._dragging_point = None
        self._points = {0: 0, 1: 1}  # Fixing points (0,0) and (1,1)

        self._init_plot()

    def _init_plot(self):
        self._figure = plt.figure("Example plot")
        self._axes = plt.subplot(1, 1, 1)
        self._axes.set_xlim(0, 1)
        self._axes.set_ylim(0, 1)
        self._axes.grid(which="both")
        
        # Fixing points (0,0) and (1,1)
        self._points = {0: 0, 1: 1}
        
        # Display the initial interpolation
        self._update_plot()
        
        # Foreground imshow
        dx, dy = 0.01, 0.01
        y_fg, x_fg = np.mgrid[slice(0, 1 + dy, dy), slice(0, 1 + dx, dx)]
        z_fg = n(y_fg)
        z_min, z_max = np.abs(z_fg).min(), np.abs(z_fg).max()
        self._foreground = self._axes.imshow(z_fg, vmin=z_min, vmax=z_max, extent=[x_fg.min(), x_fg.max(), y_fg.min(), y_fg.max()], interpolation='nearest', origin='lower', cmap='YlOrBr')
        
        # Colorbar
        cbar = plt.colorbar(self._foreground)
        cbar.set_label(r"$n$")
        
        # Plot BVP Solution
        self._axes.plot(sol_bvp.x, sol_bvp.y[0], color='green')#, label='BVP Solution')
        
        self._figure.canvas.mpl_connect('button_press_event', self._on_click)
        self._figure.canvas.mpl_connect('button_release_event', self._on_release)
        self._figure.canvas.mpl_connect('motion_notify_event', self._on_motion)
        plt.xlabel(r"$x$ in m")
        plt.ylabel(r"$y$ in m")
      #  plt.legend()
        plt.show()
    
    def _update_plot(self):
        if not self._points:
            self._line.set_data([], [])
        else:
            x, y = zip(*sorted(self._points.items()))
            # Sort points
            sorted_indices = np.argsort(x)
            x_sorted = np.array(x)[sorted_indices]
            y_sorted = np.array(y)[sorted_indices]
            # Interpolate using cubic spline
            cs = CubicSpline(x_sorted, y_sorted)
            x_interp = np.linspace(min(x_sorted), max(x_sorted), 10000)
            y_interp = cs(x_interp)
            # Calculate runtime
            runtime = 0
            outside=False
            for i in range(len(x_interp) - 1):
                if not (0<=y_interp[i]<=1 and 0<=x_interp[i]<=1):
                    outside=False
                runtime += n((y_interp[i + 1] + y_interp[i])/2) *math.sqrt((x_interp[i + 1] - x_interp[i])**2 + (y_interp[i + 1] - y_interp[i])**2)
            # Plot spline
            if not self._line:
                self._line, = self._axes.plot(x_interp, y_interp, "b")
            else:
                self._line.set_data(x_interp, y_interp)
            # Update runtime text field
            if hasattr(self, "_runtime_text"):
                self._runtime_text.remove()
            if outside == True:
                self._runtime_text = self._axes.text(0.05, 0.95, f"Lichtstrahl außerhalb des Mediums", transform=self._axes.transAxes, ha='left', va='top', color='white', fontsize=12, bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 5})
            if outside == False:
                self._runtime_text = self._axes.text(0.05, 0.95, f"Laufzeit: {runtime:.5f} s", transform=self._axes.transAxes, ha='left', va='top', color='white', fontsize=12, bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 5})
        # Update scatter plot for points
        if self._scatter:
            self._scatter.remove()
        x_scatter, y_scatter = zip(*self._points.items())
        self._scatter = self._axes.scatter(x_scatter, y_scatter, color='red', marker='o')
        self._figure.canvas.draw()
        
        



    def _add_point(self, x, y=None):
        if isinstance(x, MouseEvent):
            x, y = x.xdata, x.ydata
        self._points[x] = y
        return x, y

    def _remove_point(self, x, _):
        if x in self._points:
            self._points.pop(x)

    def _find_neighbor_point(self, event):
        u""" Find point around mouse position

        :rtype: ((int, int)|None)
        :return: (x, y) if there are any point around mouse else None
        """
        distance_threshold = 0.05
        nearest_point = None
        min_distance = math.sqrt(2 * (1 ** 2))
        for x, y in self._points.items():
            distance = math.hypot(event.xdata - x, event.ydata - y)
            if distance < min_distance:
                min_distance = distance
                nearest_point = (x, y)
        if min_distance < distance_threshold:
            return nearest_point
        return None

    def _on_click(self, event):
        u""" callback method for mouse click event

        :type event: MouseEvent
        """
        # left click
        if event.button == 1 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            else:
                self._add_point(event)
            self._update_plot()
        # right click
        elif event.button == 3 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._remove_point(*point)
                self._update_plot()

    def _on_release(self, event):
        u""" callback method for mouse release event

        :type event: MouseEvent
        """
        if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point:
            self._dragging_point = None
            self._update_plot()

    def _on_motion(self, event):
        u""" callback method for mouse motion event

        :type event: MouseEvent
        """
        if not self._dragging_point:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._remove_point(*self._dragging_point)
        self._dragging_point = self._add_point(event)
        self._update_plot()



