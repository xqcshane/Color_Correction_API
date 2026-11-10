from __future__ import annotations

import numpy as np

from api.gpu_probe import concise_gpu_error, run_cupy_kernel_probe


class _Stream:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class _Cuda:
    def __init__(self):
        self.stream = _Stream()

    def get_current_stream(self):
        return self.stream


class _FakeCuPy:
    float32 = np.float32
    arange = staticmethod(np.arange)
    where = staticmethod(np.where)
    sum = staticmethod(np.sum)
    asnumpy = staticmethod(np.asarray)

    def __init__(self):
        self.cuda = _Cuda()


def test_real_kernel_probe_executes_and_synchronizes():
    fake = _FakeCuPy()

    run_cupy_kernel_probe(fake)

    assert fake.cuda.stream.synchronized is True


def test_gpu_error_is_single_line_and_bounded():
    error = RuntimeError("line one\n" + "x" * 1000)
    message = concise_gpu_error(error, limit=80)

    assert "\n" not in message
    assert message.startswith("RuntimeError:")
    assert len(message) <= 94
