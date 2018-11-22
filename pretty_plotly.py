# pretty plot with plotly
from plotly.offline import init_notebook_mode, iplot, plot
from plotly.graph_objs import *
from numpy import arange

def unsqueeze(val):
    '''
    Change the shape of single dimensional numpy array from (N,) to (N,1) 
    
    Parameters
    ----------
    val : numpy array with dimension (N,)
    
    Returns
    -------
    numpy array
    
    '''
    return val.reshape(len(val), -1)  

init_notebook_mode(connected=True)   
def line_plotly(y, n=[], x=[], xTitle ='', yTitle= '', title='', outFileName=[]):
    '''
    line plot with plotly.
    The input data y, is row major, meaning each row is a new data set, and columns are samples.
    
    Parameters
    ----------
    y : ndarray (1 or 2 dimensions) 
        data to be plotted
    n : list of strings
        label for each dataset
    x : ndarray (1 or 2 dimensions) 
        index for each data set in y
    xTitle : string
        text label for x axis
    yTitle : string
        text label for y axis
    title : string
        text label for title of plot
    outFileName : string
        if used plot is saved to file, otherwise goes to standard out
        
    '''
    bNoX = len(x) == 0
    bNoN = len(n) == 0
    data =[]
    
    if len(y.shape) == 1:
        y = unsqueeze(y).T
        
    for ii, val in enumerate(y):
        
        # deal with defaults
        if bNoX:
            xVal = arange(len(val))
        else:    
            xVal = x[ii]
            
        if bNoN:
            nVal = ''
        else:    
            nVal = n[ii]

        data.append(Scatter(y=val, x=xVal, name=nVal))

    layout = Layout(
        title = title,
        autosize=True,
        xaxis=dict(
            title=xTitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=yTitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )       
    )
    
    fig = Figure(data=data, layout=layout)

    if len(outFileName) > 0:
        init_notebook_mode(connected=False)   
        plot(fig, filename=outFileName)
    else:
        init_notebook_mode(connected=True)   
        iplot(fig)   
        
def scatter_plotly(y, x, n=[], xTitle ='', yTitle= '', title='', outFileName=[]):
    '''
    scatter plot in 2 dimensions with plotly
    The input data y, is row major, meaning each row is a new data set, and columns are samples.

    Parameters
    ----------
    y : ndarray (1 or 2 dimensions) / list, list of lists
        data to be plotted
    x : ndarray (1 or 2 dimensions) / list, list of lists
        data to be plotted
    n : list of strings
        label for each dataset
    xTitle : string
        text label for x axis
    yTitle : string
        text label for y axis
    title : string
        text label for title of plot
    outFileName : string
        if used plot is saved to file, otherwise goes to standard out
        
    '''    
    if len(y.shape) == 1:
        y = unsqueeze(y).T
        x = unsqueeze(x).T
        
    bNoN = len(n) == 0
    data =[]
    
    mode = 'markers'
    marker= dict(size= 9,
            line= dict(width=1),
            opacity= 0.3
            )
    
    for ii, val in enumerate(y):
        
        # deal with defaults            
        if bNoN:
            nVal = ''
        else:    
            nVal = n[ii]

        data.append(Scatter(y=val, x=x[ii], name=nVal, mode=mode, marker=marker))

    layout = Layout(
        title = title,
        autosize=True,
        xaxis=dict(
            title=xTitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=yTitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )       
    )
    
    fig = Figure(data=data, layout=layout)

    if len(outFileName) > 0:
        init_notebook_mode(connected=False)   
        plot(fig, filename=outFileName)
    else:
        init_notebook_mode(connected=True)   
        iplot(fig)           

def hist_plotly(x, n=[], xTitle ='', yTitle= '', title='', outFileName=[]):
    '''
    histogram with plotly
    
    Parameters
    ----------
    x : ndarray (1 or 2 dimensions) / list, list of lists
        data to be histogrammed
    n : list of strings
        label for each dataset
    xTitle : string
        text label for x axis
    yTitle : string
        text label for y axis
    title : string
        text label for title of plot
    outFileName : string
        if used plot is saved to file, otherwise goes to standard out
        
    '''    
    bNoN = (len(n) == 0)
    data = []
    
    if len(x.shape) == 1:
        x = unsqueeze(x).T
        
    for ii, val in enumerate(x):
        # deal with defaults
        if bNoN:
            nVal = ''
        else:    
            nVal = n[ii]

        data.append(Histogram(x=val, name=nVal))

    layout = Layout(
        title = title,
        autosize=True,
        xaxis=dict(
            title=xTitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=yTitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )       
    )
    
    fig = Figure(data=data, layout=layout)

    if len(outFileName) > 0:
        init_notebook_mode(connected=False)   
        plot(fig, filename=outFileName)
    else:
        init_notebook_mode(connected=True)   
        iplot(fig) 
        
from numpy import asarray, squeeze, mean, std   

def plot_bland_altman(data1, data2, yTitle='measurement difference [mmHg]', xTitle='measurement average [mmHg]', title=''):
    '''
    bland altman with plotly
    
    Parameters
    ----------
    data1 : ndarray (1 or 2 dimensions) / list, list of lists
        data to be plotted
    data2 : ndarray (1 or 2 dimensions) / list, list of lists
        data to be plotted
    xTitle : string
        text label for x axis
    yTitle : string
        text label for y axis
    title : string
        text label for title of plot        
    '''
    data1     = asarray(data1)
    data2     = asarray(data2)
    avg       = squeeze(mean([data1, data2], axis=0))
    diff      = squeeze(data1 - data2)       # Difference between data1 and data2
    md        = mean(diff)                   # Mean of the difference
    sd        = std(diff, axis=0)            # Standard deviation of the difference
    deltaX    = max(avg) - min(avg)
    deltaY    = max(diff) - min(diff)
    upper     = md + 1.96*sd
    lower     = md - 1.96*sd
    abscissa = [min(avg), max(avg)]
            
    data =[]  
    mode = 'markers'
    marker= dict(size= 9,
                 line= dict(width=1),
                 opacity= 0.3)
    line = dict(color = ('rgb(205, 12, 24)'),
                width = .75,
                dash = 'dash')
    data.append(Scatter(y=diff, x=avg, 
                        mode=mode, 
                        marker=marker
                ))
    data.append(Scatter(y=[md, md], 
                        x=abscissa,
                        mode='lines+text',
                        text=['', 'mean = %0.2f'%md],
                        textposition='top'                    
                        ))
    data.append(Scatter(y=[upper, upper], 
                        x=abscissa, 
                        mode='lines+text',
                        line=line
                       ))
    data.append(Scatter(y=[lower, lower], 
                        x=abscissa, 
                        mode='lines+text',
                        line=line
                       ))

    #print(data) 
    outFileName=[]
    layout = Layout(
        title = title,
        autosize=True,
        showlegend=False,
        xaxis=dict(
            title=xTitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=yTitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )       
    )
    
    fig = Figure(data=data, layout=layout)

    if len(outFileName) > 0:
        init_notebook_mode(connected=False)   
        plot(fig, filename=outFileName)
    else:
        init_notebook_mode(connected=True)   
        iplot(fig)        
        
#def fit_hist_plotly(x, binNum, componentNum =3, samplesNum=1000, n=[], xTitle ='', yTitle= '', title='', outFileName=[]):
#    bNoN = (len(n) == 0)
#    data = []
#    
#    if x.shape[1] == 1:
#        x = x.T
#            
#    for ii, val in enumerate(x):
#        # deal with defaults
#        if bNoN:
#            nVal = ''
#            showLegend = False
#        else:    
#            nVal = n[ii]
#            showLegend = True
#
#        xbins=dict(start=min(val),end=max(val),size=(max(val)-min(val))/binNum)
#        data.append(Histogram(x=val, name=nVal, xbins=xbins, showlegend=showLegend))
#
#        temp = gmm_fit_hist(val, numBins=binNum, numComponents=componentNum, numSamples=samplesNum)
#        data.append(Scatter(y=temp[0]*temp[1], x=np.squeeze(temp[2]), name=nVal+' fit', \
#                            showlegend=showLegend, mode='lines+markers'))
# 
#     layout = Layout(
#     title = title,
#     autosize=True,
#     xaxis=dict(
#         title=xTitle,
#         titlefont=dict(
#             family='Courier New, monospace',
#             size=18,
#             color='#7f7f7f'
#         )
#     ),
#     yaxis=dict(
#         title=yTitle,
#         titlefont=dict(
#             family='Courier New, monospace',
#             size=18,
#             color='#7f7f7f'
#         )
#     )       
#     )
#     
#     fig = Figure(data=data, layout=layout)
# 
#     if len(outFileName) > 0:
#         init_notebook_mode(connected=False)   
#         plot(fig, filename=outFileName)
#     else:
#         init_notebook_mode(connected=True)   
#         iplot(fig)                 

#
#from sklearn.mixture.gaussian_mixture import GaussianMixture
#def gmm_fit_hist(data, numBins=100, numComponents=3, numSamples=1000, disp=False):
#    data = unsqueeze(data)
#    gmm = GaussianMixture(n_components=numComponents).fit(data)
#    x = unsqueeze(np.linspace(min(data), max(data), numSamples))
#    logprob = gmm.score_samples(x)
#    pdf = np.exp(logprob)
#    if disp:
#        temp = plt.hist(data, numBins, normed=False)
#        (counts, bins, notes) = temp
#    else:
#        temp = np.histogram(data, numBins, normed=False)
#        (counts, bins) = temp
#        
#    gain = np.sum(counts)*(np.diff(bins)[0])
#
#    if disp:
#        plt.plot(x, pdf*gain, '-k')
#        plt.xlabel('$x$')
#        plt.ylabel('$p(x)$')    
#    return pdf, gain, x, counts, bins                

