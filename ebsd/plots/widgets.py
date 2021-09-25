from skimage import color


w = {
    'buttons': [],
    'lineEdit': [],
    'canvas': {'canvas1': [], 'mdf': [], 'kam': [], 'disloc': [], 'grod': [], 'ipf': [], 'canvas3d': [], 'pf': [], 'ipf': []},
    'msgs': [],
    'dropdown': [],
    'pfchoice': [],
    'selpoints': [],
    'pfpoints': [],
    'recSegment':[],
    'dfs': [],
    'dx': [],
    'dy': [],
    'Binstances': []
}


def clear_widgets(widgets, widget='all'):
    ''' hide all existing widgets and erase
        them from the global dictionary'''
    if(widget == 'all'):
        for widget in widgets:
            if widgets[widget] != [] and widget != 'canvas':
                widgets[widget].clear()
        for i in range(0, len(widgets[widget])):
            widgets[widget].pop()

    elif widget == 'canvas':
        for item in widgets['canvas']:
            widgets['canvas'][item].clear()
    else:
        for i in range(0, len(widgets[widget])):
            widgets[widget].pop()
