import os

def retrieve_file_type(file_path, file_prefix='fly', file_type='.h5'):
    path = os.path.abspath(file_path)
    files = sorted(os.listdir(file_path))
    files_filted = []
    n_type = len(file_type)
    n_start = len(file_prefix)
    for f in files:
        if f[-n_type:] == file_type and f[:n_start] == file_prefix:
            f = f'{path}/{f}'
            files_filted.append(f)
    return files_filted

def extract_range(text, datatype='int'):
    if text == '[]':
        return []
    if text[0] == '[' and text[-1] == ']':
        tx = text[1:-1]
        tx = tx.replace(' ', '')
        tx = tx.split(',')
        if datatype == 'float':
            return [float(tx[0]), float(tx[-1])]
        elif datatype == 'int':
            return [int(tx[0]), int(tx[-1])]
    else:
        return []