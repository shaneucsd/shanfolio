def convert_symbol(x):
    # if x can be converted to int
    try:
        x = str(int(x)).zfill(6)
    except:
        pass
    return x
