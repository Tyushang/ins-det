# All torch.cuda.memory_stats keys.
# 'active.all.allocated',
# 'active.all.current',
# 'active.all.freed',
# 'active.all.peak',
# 'active.large_pool.allocated',
# 'active.large_pool.current',
# 'active.large_pool.freed',
# 'active.large_pool.peak',
# 'active.small_pool.allocated',
# 'active.small_pool.current',
# 'active.small_pool.freed',
# 'active.small_pool.peak',
# 'active_bytes.all.allocated',
# 'active_bytes.all.current',
# 'active_bytes.all.freed',
# 'active_bytes.all.peak',
# 'active_bytes.large_pool.allocated',
# 'active_bytes.large_pool.current',
# 'active_bytes.large_pool.freed',
# 'active_bytes.large_pool.peak',
# 'active_bytes.small_pool.allocated',
# 'active_bytes.small_pool.current',
# 'active_bytes.small_pool.freed',
# 'active_bytes.small_pool.peak',
# 'allocated_bytes.all.allocated',
# 'allocated_bytes.all.current',
# 'allocated_bytes.all.freed',
# 'allocated_bytes.all.peak',
# 'allocated_bytes.large_pool.allocated',
# 'allocated_bytes.large_pool.current',
# 'allocated_bytes.large_pool.freed',
# 'allocated_bytes.large_pool.peak',
# 'allocated_bytes.small_pool.allocated',
# 'allocated_bytes.small_pool.current',
# 'allocated_bytes.small_pool.freed',
# 'allocated_bytes.small_pool.peak',
# 'allocation.all.allocated',
# 'allocation.all.current',
# 'allocation.all.freed',
# 'allocation.all.peak',
# 'allocation.large_pool.allocated',
# 'allocation.large_pool.current',
# 'allocation.large_pool.freed',
# 'allocation.large_pool.peak',
# 'allocation.small_pool.allocated',
# 'allocation.small_pool.current',
# 'allocation.small_pool.freed',
# 'allocation.small_pool.peak',
# 'inactive_split.all.allocated',
# 'inactive_split.all.current',
# 'inactive_split.all.freed',
# 'inactive_split.all.peak',
# 'inactive_split.large_pool.allocated',
# 'inactive_split.large_pool.current',
# 'inactive_split.large_pool.freed',
# 'inactive_split.large_pool.peak',
# 'inactive_split.small_pool.allocated',
# 'inactive_split.small_pool.current',
# 'inactive_split.small_pool.freed',
# 'inactive_split.small_pool.peak',
# 'inactive_split_bytes.all.allocated',
# 'inactive_split_bytes.all.current',
# 'inactive_split_bytes.all.freed',
# 'inactive_split_bytes.all.peak',
# 'inactive_split_bytes.large_pool.allocated',
# 'inactive_split_bytes.large_pool.current',
# 'inactive_split_bytes.large_pool.freed',
# 'inactive_split_bytes.large_pool.peak',
# 'inactive_split_bytes.small_pool.allocated',
# 'inactive_split_bytes.small_pool.current',
# 'inactive_split_bytes.small_pool.freed',
# 'inactive_split_bytes.small_pool.peak',
# 'num_alloc_retries',
# 'num_ooms',
# 'reserved_bytes.all.allocated',
# 'reserved_bytes.all.current',
# 'reserved_bytes.all.freed',
# 'reserved_bytes.all.peak',
# 'reserved_bytes.large_pool.allocated',
# 'reserved_bytes.large_pool.current',
# 'reserved_bytes.large_pool.freed',
# 'reserved_bytes.large_pool.peak',
# 'reserved_bytes.small_pool.allocated',
# 'reserved_bytes.small_pool.current',
# 'reserved_bytes.small_pool.freed',
# 'reserved_bytes.small_pool.peak',
# 'segment.all.allocated',
# 'segment.all.current',
# 'segment.all.freed',
# 'segment.all.peak',
# 'segment.large_pool.allocated',
# 'segment.large_pool.current',
# 'segment.large_pool.freed',
# 'segment.large_pool.peak',
# 'segment.small_pool.allocated',
# 'segment.small_pool.current',
# 'segment.small_pool.freed',
# 'segment.small_pool.peak',

import pandas as pd


def format_number(n_bytes):
    assert n_bytes >= 0

    string_list = []

    number_to_format = n_bytes // 1024 // 1024
    while number_to_format > 0:
        high, low = divmod(number_to_format, 1024)
        if high > 0:
            string_list = [f'{low:03d}'] + string_list
        else:
            string_list = [f'{low}'] + string_list

        number_to_format = high

    return ','.join(string_list)


cols = [
    'active_bytes.all.current',
    'active_bytes.all.peak',
    'allocated_bytes.all.current',
    'allocated_bytes.all.peak',
    'inactive_split_bytes.all.current',
    'inactive_split_bytes.all.peak',
    'reserved_bytes.all.current',
    'reserved_bytes.all.peak',
]

raw_df = pd.read_csv('./men_stats_4.csv', usecols=['memo'] + cols, index_col='memo')

# ms_df = raw_df.applymap(format_number)

ms_df = raw_df.applymap(lambda x: x // 1024 // 1024 / 1024)

