import numpy as np
from datetime import datetime
from pickle import dump, load, HIGHEST_PROTOCOL

def printnow(to_print):
    '''
    Flush output buffer and print to standard out. This will cause print to happen immediately. 
    
    Parameters
    ----------
    to_print : formatted string
    
    Example
    -------
    r = 1
    printnow('The value of r is %d'%r)
    >>> The value of r is 1
        
    '''
    print(to_print, flush=True)
    
    
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

def now(formatDate='%Y-%m-%d'):
    '''
    Returns the datime and time when executed in specified format
    
    Parameters
    ----------
    formatDate : string format for date time
    
    Returns
    -------
    datetime
    
    Example
    -------
    now()
    >>> '2018-06-01'

    now(formatDate='%Y-%m-%d %H:%M:%S')  
    >>> '2018-06-01 13:11:31'

    '''
    return datetime.now().strftime(formatDate)


def read_from_pickle(path):
    '''
    Read pickle file. 
    
    Returns a list of all variables stored in file.
    
    Parameters
    ----------
    path : path and filename

    '''

    objects = []
    with open(path, 'rb') as file:
        while True:
            try:
                objects.append(load(file))
            except EOFError:
                break
    return objects           


def save_to_pickle(path, data):
    '''
    Save variables to pickle file. Put multiple variables in a list. 
    
    Parameters
    ----------
    path : directory structure and filename to save
    data : list of variables

    Example
    -------
    save_to_picke('//nas2.valencell.com/Shared/Data/example.pkl', [variable_1, dataframe_0, dictionary_4])
    '''

    with open(path, 'wb') as handle:
        dump(data, handle, protocol=HIGHEST_PROTOCOL)


def in_list(first, second):
    '''
    Find all the elements of first list that matches any element in second list.
    
    Returns a list of booleans the same size as first list
    
    Parameters
    ----------
    path : directory structure and filename to save
    data : list of variables
    
    Returns
    -------
    list

    Example
    -------
    in_list(['a','b','c','d', 'c'], ['b', 'c'])    
    > [False, True, True, False, True]
    '''
    return list(np.compress(first, np.in1d(first,second)))


def has_digit(inputString):
    '''
    Boolean check if the string contains a digit in any character.

    Parameters
    ----------
    inputString : string to interrogate
    
    Returns
    -------
    boolean
    
    See Also
    --------
    np_has_digit which is a vectorized version of this for arrays
    '''
    return any(char.isdigit() for char in str(inputString))

# figure this out
np_has_digit = np.vectorize(has_digit, otypes=[np.bool])


def make_columns_unique(df_columns):
    '''
    Create unique column names for a data frame. Non-unique names will be appended with an underscore and integer. 
    
    Parameters
    ----------
    list : df_columns
    
    Example
    -------
    df_columns = ['a', 'b', 'a', 'a_2', 'a_2', 'a', 'a_2', 'a_2_2']
    list(uniquify(df_columns))
    >>> ['a', 'b', 'a_2', 'a_2_2', 'a_2_3', 'a_3', 'a_2_4', 'a_2_2_2']
    '''        
    seen = set()

    for item in df_columns:
        fudge = 1
        newitem = item

        while newitem in seen:
            fudge += 1
            newitem = "{}_{}".format(item, fudge)

        yield newitem
        seen.add(newitem)
    return seen    

def is_str_in_df(df, string):
    '''
    Search all dataframe cells for a specific string
    
    Parameters
    ----------
    DataFrame : df
    string : string
    
    Returns
    -------
    Boolean if any cell contains string, Boolen of indexes for each cell
    
    '''    
    rows, cols = df.shape
    find = np.zeros([rows, cols])
    for col in np.arange(cols):
        find[:,col] = df.iloc[:,col].astype(str).str.lower().str.contains(string).values 
    # return boolean and indices    
    return(np.any(find), np.nonzero(find))    

def is_array_of_type(array, varType):
    '''
    Checks each member of array againts given varialbe type.
    
    Parameters
    ----------
    array : numpy array to interrogate
    varType : variable type
    
    Returns
    -------
    numpy boolean array
    
    Example
    -------
    is_array_of_type(temp, np.int32)
    '''
    return np.array([isinstance(val, varType) for val in array])

def has_three_consecutive_int(val):
    '''
    Check if an alphanumeric varialbe has three consecutive integers.
    
    Parameters
    ----------
    string like : val
    
    Returns
    -------
    boolean
    
    Example
    -------
    has_three_consecutive_int('agh783jght)
    >>> True
    '''
    val = str(val)
    isInt = [char.isdigit() for char in val]
    tupleOfThree = (list(zip(isInt,isInt[1:],isInt[2:])))
    return np.array(any([all(val) for val in tupleOfThree]))
