import numpy as np

from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import figure


def tropicalize(points: np.ndarray, base: int=np.e):
    abs_points = np.abs(points)
    log_points = np.log(points) / np.log(base)

    return log_points


def initial_data(m: int=2, b: int=1, base: int=10,
                 num_points: int=10**5+1, 
                 x_min: float=-100, x_max: float=100):
    x = np.linspace(-100, 100, 10**5).reshape((-1, 1))
    y = m*x + b
    trop_points = tropicalize(np.hstack([x, y]), base=base)

    return ColumnDataSource(data=dict(x=x, y=y, trop_x=trop_points[:,0], trop_y=trop_points[:,1]))


def main():
    base_slider = Slider(start=2, end=100, value=2, step=1, title='base')
    
    m, b = 2, 1
    source = initial_data(m=m, b=b, base=base_slider.value)
    x_min, x_max = np.nanmin(source.data['trop_x']), np.nanmax(source.data['trop_x'])
    y_min, y_max = np.nanmin(source.data['trop_y']), np.nanmax(source.data['trop_y'])
    plot = figure(plot_width=600, plot_height=600, 
                  x_range=(x_min, x_max), y_range=(y_min, y_max),
                  title=f'y = {m}x + {b}')
    plot.line('trop_x', 'trop_y', source=source, line_width=3, line_alpha=0.6)

    update_curve = CustomJS(args=dict(source=source, slider=base_slider), code="""
        var data = source.data;
        var base = slider.value;
        var x = data['x']
        var y = data['y']
        var trop_x = data['trop_x']
        var trop_y = data['trop_y']
        for (var i = 0; i < x.length; i++) {
            trop_x[i] = Math.log(Math.abs(x[i])) / Math.log(base)
            trop_y[i] = Math.log(Math.abs(y[i])) / Math.log(base)
        }
        
        // necessary becasue we mutated source.data in-place
        source.change.emit();
    """)
    base_slider.js_on_change('value', update_curve)

    show(column(base_slider, plot))


if __name__ == '__main__':
    main()