import pandas as pd


def remove_empty_rows(frame: pd.DataFrame, labels: list | str) -> pd.DataFrame:
    if isinstance(labels, str):
        labels = [labels]

    result = frame.copy()
    column_map = {
        'lab': 'lab',
        'lcn': 'lcn',
        'lnp': 'lnp',
        'sbj': 'sbj',
        'dsc': 'dsc',
        'curi': 'curi',
        'puri': 'puri',
        'voc': 'voc',
        'tdl': 'tdl'
    }

    for label in labels:
        if label in column_map:
            column = column_map[label]
            result = result[result[column] != '']

    return result
