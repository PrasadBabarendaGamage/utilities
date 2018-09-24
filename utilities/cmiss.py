def cm_str_to_array(String):
    """
    Expands a cm or cmgui string of compressed dofs
    (eg. node number or element numbers). E.g. "1,2..4" to [1,2,3,4]
    """
    StringArray = String.split(',')
    Data = []
    for value in StringArray:
        if value.find('..') > -1:
            temp = value.split('..')
            temp2 = range(int(temp[0]), int(temp[1]) + 1)
            [Data.append(point) for point in temp2]
        else:
            Data.append(int(value))
    return Data
