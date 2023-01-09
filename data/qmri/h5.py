import h5py
import torch
import numpy as np

def to_hdf5(ds, path):
    def wf(_):
        import torch
        import numpy
        import random

        torch.manual_seed(0)
        numpy.random.seed(0)
        random.seed(0)

    dl = torch.utils.data.DataLoader(ds, worker_init_fn=wf, num_workers=1, batch_size=1, shuffle=False)
    with h5py.File(path, "x") as outfile:
        for i, data in enumerate(dl):
            for j, d in enumerate(data):
                outfile[f"/{i}/{j}"] = np.array(d[0])
            outfile[f"/{i}"].attrs["max"] = j
        outfile.attrs["max"] = i


class hdf5DS(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(self.path, "r") as f:
            self._len = int(f.attrs["max"]) + 1

    def __getitem__(self, idx):
        if idx >= self._len or idx < -self._len:
            raise IndexError("index out of range")
        i = self._len + idx if idx < 0 else idx
        ret = []
        with h5py.File(self.path, "r") as f:
            ds = f[f"/{i}"]
            for j in range(ds.attrs["max"] + 1):
                d = ds[f"{j}"]
                ret.append(np.array(d))
        return tuple(ret)

    def __len__(self):
        return self._len

    
def h5append(file, key, data, chunks=None, compression='lzf'):
    data = np.atleast_1d(np.array(data))
    if np.asarray(data).dtype.kind == 'U':
        data = np.asarray(data).astype(h5py.string_dtype(encoding='ascii'))

    if isinstance(file, h5py.Dataset):
        if not (key is None or key == '/'):
            raise KeyError('If file is a dataset, key must be /')
        else:
            ds = file
    else:
        if key not in file.keys():
            file.create_dataset(
                key,
                chunks=chunks or data.shape,
                compression=compression,
                shuffle=True if compression else None,
                data=data,
                maxshape=(None, *data.shape[1:]),
            )
            return
        else:
            ds = file[key]
    ds.resize((ds.shape[0] + data.shape[0]), axis=0)
    ds[-data.shape[0] :] = data