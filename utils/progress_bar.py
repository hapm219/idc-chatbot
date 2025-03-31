from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

def run_parallel_pipeline(
    items,
    func,
    shared_args=(),
    max_processes=8,
    desc="🔄 Đang xử lý song song",
    ncols=100
):
    """
    Chạy song song xử lý danh sách `items` với progress bar cập nhật theo thời gian thực.
    """
    with Pool(processes=max_processes) as pool:
        wrapped_func = partial(func, *shared_args)
        with tqdm(total=len(items), desc=desc, ncols=ncols) as pbar:
            for _ in pool.imap_unordered(wrapped_func, items):
                pbar.update(1)

def run_sequential_pipeline(
    items,
    func,
    shared_args=(),
    desc="🔁 Đang xử lý tuần tự",
    ncols=100
):
    """
    Chạy tuần tự danh sách `items` với progress bar.
    """
    wrapped_func = partial(func, *shared_args)
    for item in tqdm(items, desc=desc, ncols=ncols):
        wrapped_func(item)
