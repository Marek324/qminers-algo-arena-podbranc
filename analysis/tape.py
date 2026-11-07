"""Reading a backtest tape (the simulator's CSV format) into a DataFrame."""

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

INSERT = 'Insert'
CANCEL = 'Cancel'


def load_tape(path: Path | IO[bytes]) -> pd.DataFrame:
    """Load a tape CSV (plain or .gz) with the columns the simulator reads.

    Adds `ts_us` (int64 microseconds, rounded the same way as the simulator),
    `is_insert`, `side_sign` (+1 buy, -1 sell, 0 for cancels) and `ioc`.
    """
    df = pd.read_csv(
        path,
        dtype={
            'Time': 'float64',
            'Operation': 'string',
            'OrderId': 'string',
            'Side': 'string',
            'Volume': 'Int64',
            'Price': 'Int64',
            'Lifespan': 'string',
        },
    )
    unknown = set(df['Operation'].dropna().unique()) - {INSERT, CANCEL}
    if unknown:
        raise ValueError(f'Unsupported operations in {path}: {sorted(unknown)}')
    df['ts_us'] = np.round(df['Time'].to_numpy() * 1_000_000).astype('int64')
    df['is_insert'] = (df['Operation'] == INSERT).to_numpy(dtype=bool)
    side = df['Side'].fillna('')
    df['side_sign'] = np.where(side == 'B', 1, np.where(side == 'A', -1, 0)).astype('int8')
    if 'Lifespan' in df.columns:
        df['ioc'] = (df['Lifespan'].fillna('') == 'F').to_numpy(dtype=bool) & df['is_insert'].to_numpy()
    else:
        df['ioc'] = False
    return df
