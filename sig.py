import numpy as np
from scipy import signal as sg
from numpy import linalg as lg


def slope_intercept(x,y):
    '''
    slope_intercept takes x and y coordinates of a data set and return a tuple of slope & intercept across the specified points.
    
    Parameters
    ----------
    x : ndarray of size N
        x samples
    y : ndarray of size N
        y samples
        
    Returns
    -------
    slope : float
        the slope of the input data
    intercept : float
        the intercept of the input data

    '''
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return (slope, intercept)



def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    butter_bandpass creates a set transfer function coefficients of an nth-order bandpass digital Butterworth filter

    Parameters
    ----------
    lowcut : int/float
        Lowcut corner point frequency
    highcut : int/float
        Highcut corner point frequency
    fs : int
       The sampling frequency  [Hz]
    order : int
       N-th filter order

    Returns
    -------
    b, a : ndarray, ndarray
        Transfer function coefficients of an nth-order bandpass digital Butterworth filter

    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(lowcut, fs, order=5):
    '''
    butter_lowpass creates a set transfer function coefficients of an nth-order lowpass digital Butterworth filter

    Parameters
    ----------
    lowcut : int/float
        Lowcut corner point frequency
    fs : int
       The sampling frequency [Hz]
    order : int
       N-th filter order

    Returns
    -------
    b, a : ndarray, ndarray
        Transfer function coefficients of an nth-order lowpass digital Butterworth filter

    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a

def butter_highpass(highcut, fs, order=5):
    '''
    butter_highpass creates a set transfer function coefficients of an nth-order highpass digital Butterworth filter

    Parameters
    ----------
    highcut : int/float
        Highcut corner point frequency
    fs : int
       The sampling frequency  [Hz]
    order : int
       N-th filter order

    Returns
    -------
    b, a : ndarray, ndarray
        Transfer function coefficients of an nth-order highpass digital Butterworth filter

    '''
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='high')
    return b, a

def sliding_window(data, size=2, step=1, fillvalue=None): #CHANGE
    '''
    Parameters
    ----------
    iterable : ndarray
        data to move sliding window along
    size : int
        Window size in number of samples
    step : int
        Step size in number of samples to stride across signal

    Returns
    -------
    sig : iterable

    '''
    if size < 0 or step < 1:
        raise ValueError
    it = iter(data)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration: # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))

def peaks(sig):
    """
    detects peaks and valleys as defined by the first derivative changing signs
    0  : no peak
    2  : peak second derivative positve
    -2 : peak second derivative negative

    Parameters
    ----------
    signal : ndarray 
        input signal to be analyzed

    Returns
    -------
    peak : ndarray
        list of indicies corresponding to peaks/valleys found in the time-domain of the signal
    """

    peak = signal.lfilter([-1, 1], [1], np.sign(np.diff(sig)))
    peak[0] = 0
    peak = np.append(peak, [0])
    return peak

def pseudospectrum_MUSIC(x,L,M=None,Fs=1,f=None):
    """
    This function compute the MUSIC pseudospectrum. The pseudo spectrum is defined as
    math:: S(f)=\frac{1}{\|\textbf{G}^{H}\textbf{a}(f) \|}
    where :math:`\textbf{G}` corresponds to the noise subspace and :math:`\textbf{a}(f)` is the steering vector. The peak locations give the frequencies of the signal.
        
    Parameters
    ----------
    x : ndarray of size N
        Signal to be analyzed
    L: int
        Number of components to be extracted.
    M: int, optional. 
        Size of signal block.
    Fs: int/float
        Sampling frequency.
    f: nd array
        Frequency locations f where the pseudo spectrum is evaluated.
    Returns
    ----------
    f: ndarray
        Array of equally spaced frequncies
    cost: ndarray
    The cost function of the signal
    
        
        >>> from pylab import *
        >>> import numpy as np
        >>> import spectral_analysis.spectral_analysis as sa
        >>> Fs=500
        >>> t=1.*np.arange(100)/Fs
        >>> x=np.exp(2j*np.pi*55.2*t)
        >>> f,P=sa.pseudospectrum_MUSIC(x,1,100,Fs,None)
        >>> plot(f,P)
        >>> show()
        
        """
    
    # length of the vector x
    N=x.shape[0]
    
    if np.any(f)==None:
        f=np.linspace(0.,Fs//2,512)

    if M==None:
        M=N//2

    #extract noise subspace
    R=compute_autocovariance(x,M)
    U,S,V=lg.svd(R)
    G=U[:,L:]

    #compute MUSIC pseudo spectrum
    N_f=f.shape
    cost=np.zeros(N_f)
    
    for indice,f_temp in enumerate(f):
        # construct a (note that there a minus sign since Yn are defined as [y(n), y(n-1),y(n-2),..].T)
        vect_exp=-2j*np.pi*f_temp*np.arange(0,M)/Fs
        a=np.exp(vect_exp)
        a=np.transpose(np.matrix(a))
        #Cost function
        cost[indice]=1./lg.norm((G.H)*a)

    return f,cost

def music_spec(sig, Fs, windowSize, stepSize, ORDER):
    """
    Calculate the MUSIC spectrum of signal (sig) estimating the largest ORDER number of frequency components. MUSIC is calculated on window of windowSize samples with stepSize stride length.
    
    Parameters
    ----------
    sig : ndarray 
        Input signal to be analyzed
    Fs : int/float
        Sampling frequency
    dLen : int/float
        Window length to analyze (in seconds)
    sSize :
        Step size (in seconds)
    ORDER :
        Number of peaks to be extracted

    Returns
    -------
    t: ndarray
        time points corresponding to the size of the signal (mostly for plotting)
    f: ndarray
        requency points corresponding to frequency components of the signal (mostly for plotting)
    out: ndarray
        MUSIC cost function for each window in the signal
    Sxx1Norm: ndarray
        Cost function normalized by maximum cost in each column 

    """
    iData = sliding_window(sig, dLen, sSize, fillvalue=0)
    out = np.zeros([512, np.int(len(sig)/sSize)]) #this is hardwired in pseudospectrum_MUSIC
    count = 0
    for win in iData:
        data = [ii for ii in win]
        #data = data - np.mean(data, axis=0)
        pfreq,p = pseudospectrum_MUSIC(np.array(data),ORDER,None,Fs,None)
        out[:,count] = p  
        count = count + 1
    
    maxSpec1Cols = np.amax(out, axis=0)
    maxSpec1Cols[maxSpec1Cols==0] = 1e-6
    Sxx1Norm = out / maxSpec1Cols
    
    f = np.linspace(0, Fs/2, num=out.shape[0])
    t = np.arange(out.shape[1])
    return t, f, out, Sxx1Norm


def compute_autocovariance(x,M):
    
    '''
    This function compute the auto-covariance matrix of a numpy signal. The auto-covariance is computed as follows
        
    .. math:: \textbf{R}=\frac{1}{N}\sum_{M-1}^{N-1}\textbf{x}_{m}\textbf{x}_{m}^{H}

    where :math:`\textbf{x}_{m}^{T}=[x[m],x[m-1],x[m-M+1]]`.

    Parameters
    ----------
    x : ndarray of size N
        Signal to be analyzed

    M: int, optional. 
        Size of signal block.

    Returns
    ----------
    R: ndarray
        Autocovariance matrix
        
    '''
    
    # Create covariance matrix for psd estimation
    # length of the vector x
    N=x.shape[0]
    
    #Create column vector from row array
    x_vect=np.transpose(np.matrix(x))
    
    # init covariance matrix
    yn=x_vect[M-1::-1]
    R=yn*yn.H
    for indice in range(1,N-M):
        #extract the column vector
        yn=x_vect[M-1+indice:indice-1:-1]
        R=R+yn*yn.H
    
    R=R/N
    return R

def compute_covariance(X):
    """This function estimate the covariance of a zero-mean numpy matrix. The covariance is estimated as :math:`\textbf{R}=\frac{1}{N}\textbf{X}\textbf{X}^{H}`

    Parameters
    ----------
    X : M*N matrix
        Matrix to be analyzed

    Returns
    ----------
    R: ndarray
        covariance matrix of size M*M
    """
        
    #Number of columns
    N=X.shape[1]
    R=(1./N)*X*X.H

    return R

def root_MUSIC(x,L,M,Fs=1):
    
    """ This function estimate the frequency components based on the roots MUSIC algorithm [BAR83]_ . The roots Music algorithm find the roots of the following polynomial
        
    .. math:: P(z)=\textbf{a}^{H}(z)\textbf{G}\textbf{G}^{H}\textbf{a}(z)

    The frequencies are related to the roots as 

    .. math:: z=e^{-2j\pi f/Fs}

    Parameters
    ----------
    x : ndarray of size N
        Signal to be analyzed
    L: int
        Number of components to be extracted.
    M: int, optional. 
        Size of signal block.
    Fs: int/float
        Sampling frequency.

    Returns
    ----------
    f: ndarray
        ndarray containing the L components

        
    """

    # length of the vector x
    N=x.shape[0]
    
    if M==None:
        M=N//2
    
    #extract noise subspace
    R=compute_autocovariance(x,M)
    U,S,V=lg.svd(R)
    G=U[:,L:]

    #construct matrix P
    P=G*G.H

    #construct polynomial Q
    Q=0j*np.zeros(2*M-1)
    #Extract the sum in each diagonal
    for (idx,val) in enumerate(range(M-1,-M,-1)):
        diag=np.diag(P,val)
        Q[idx]=np.sum(diag)

    #Compute the roots
    roots=np.roots(Q)

    #Keep the roots with radii <1 and with non zero imaginary part
    roots=np.extract(np.abs(roots)<1,roots)
    roots=np.extract(np.imag(roots) != 0,roots)

    #Find the L roots closest to the unit circle
    distance_from_circle=np.abs(np.abs(roots)-1)
    index_sort=np.argsort(distance_from_circle)
    component_roots=roots[index_sort[:L]]

    #extract frequencies ((note that there a minus sign since Yn are defined as [y(n), y(n-1),y(n-2),..].T))
    angle=-np.angle(component_roots)

    #frequency normalisation
    f=Fs*angle/(2.*np.pi)

    return f

def Esprit(x,L,M,Fs):
    
    """ This function estimate the frequency components based on the ESPRIT algorithm [ROY89]_ 
        
        The frequencies are related to the roots as :math:`z=e^{-2j\pi f/Fs}`. See [STO97]_ section 4.7 for more information about the implementation.
               
    Parameters
    ----------
    x : ndarray of size N
        Signal to be analyzed
    L: int
        Number of components to be extracted.
    M: int, optional. 
        Size of signal block.
    Fs: int/float
        Sampling frequency.

    Returns
    ----------
    f: ndarray
        ndarray containing the L components
        """

    # length of the vector x
    N=x.shape[0]
        
    if M==None:
        M=N//2

    #extract signal subspace
    R=compute_autocovariance(x,M)
    U,S,V=lg.svd(R)
    S=U[:,:L]

    #Remove last row
    S1=S[:-1,:]
    #Remove first row
    S2=S[1:,:]

    #Compute matrix Phi (Stoica 4.7.12)
    Phi=(S1.H*S1).I*S1.H*S2

    #Perform eigenvalue decomposition
    V,U=lg.eig(Phi)

    #extract frequencies ((note that there a minus sign since Yn are defined as [y(n), y(n-1),y(n-2),..].T))
    angle=-np.angle(V)
    
    #frequency normalisation
    f=Fs*angle/(2.*np.pi)
    
    return f

def sped(ppsdO, threshold=0):
    r,c = ppsdO.shape
    spedgram = np.zeros_like(ppsdO)
    for ii in np.arange(c):
        p = peaks(ppsdO[:,ii])
        
        # find magnitude of each peak
        # and remove really small peaks 
        # for noise control
        v = p * ppsdO[:,ii]
        p[v<threshold] = 0
        
        spedgram[:,ii] = p
    return spedgram