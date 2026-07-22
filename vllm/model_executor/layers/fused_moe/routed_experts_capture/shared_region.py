# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared mmap for routed-experts slot buffers."""

import contextlib
import mmap
import os
import time

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger

logger = init_logger(__name__)

_WAIT_TIMEOUT_S = 30.0


def _wait_for_file_size(
    file_descriptor: int, expected_size: int, timeout: float
) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(file_descriptor).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for shared routed-experts mmap to reach "
                f"{expected_size} bytes"
            )
        time.sleep(0.005)


class SharedRoutingRegion:
    """Scheduler-owned MAP_SHARED ndarray."""

    def __init__(
        self,
        path: str,
        shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> None:
        self.path = path
        self._dtype = np.dtype(dtype)
        self._nbytes = int(np.prod(shape)) * self._dtype.itemsize
        self.fd: int | None = None
        self.mmap_obj: mmap.mmap | None = None

        self.fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.ftruncate(self.fd, self._nbytes)
        logger.info(
            "Created routed-experts mmap %s (%.2f GB)", path, self._nbytes / 1e9
        )

        self.mmap_obj = mmap.mmap(
            self.fd,
            self._nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self.array: np.ndarray = np.frombuffer(self.mmap_obj, dtype=dtype).reshape(
            shape
        )

    def close(self) -> None:
        """Release the mapping and unlink the file. Idempotent."""
        self.array = None  # type: ignore[assignment]
        if self.mmap_obj is not None:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close routed-experts mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close routed-experts fd", exc_info=True)
            self.fd = None
        try:
            os.unlink(self.path)
            logger.info("Removed routed-experts mmap %s", self.path)
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning(
                "Failed to unlink routed-experts mmap %s", self.path, exc_info=True
            )


def shared_routing_mmap_path(instance_id: str, dp_rank: int) -> str:
    """Stable per-instance, per-DP-rank path so DP ranks never collide."""
    return f"/dev/shm/vllm_routed_experts_{instance_id}_dp{dp_rank}.mmap"


class RoutedExpertsWorkerWriter:
    """Worker-side writer for the scheduler-shared routed-experts slot buffer."""

    def __init__(
        self,
        instance_id: str,
        dp_rank: int,
        slot_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> None:
        self._path = shared_routing_mmap_path(instance_id, dp_rank)
        self._slot_shape = slot_shape
        self._dtype = np.dtype(dtype)
        self._nbytes = int(np.prod(slot_shape)) * self._dtype.itemsize
        self._fd: int | None = None
        self._mmap_obj: mmap.mmap | None = None
        self._array: np.ndarray | None = None
        self._mmap_unavailable = False

    def _ensure_mmap_attached(self) -> bool:
        if self._array is not None:
            return True
        if self._mmap_unavailable:
            return False
        try:
            file_descriptor = os.open(self._path, os.O_RDWR)
        except FileNotFoundError:
            # The scheduler creates the mmap before it accepts requests. If the
            # output worker cannot see it on the first model step, the worker is
            # in a different shared-memory namespace (normally another node).
            # Return the payload in ModelRunnerOutput so the executor's existing
            # response MessageQueue can select its remote TCP transport.
            self._mmap_unavailable = True
            logger.info_once(
                "Routed-experts mmap %s is not visible to the output worker; "
                "falling back to the ModelRunnerOutput transport",
                self._path,
            )
            return False
        _wait_for_file_size(file_descriptor, self._nbytes, _WAIT_TIMEOUT_S)
        self._fd = file_descriptor
        self._mmap_obj = mmap.mmap(
            file_descriptor,
            self._nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self._array = np.frombuffer(self._mmap_obj, dtype=self._dtype).reshape(
            self._slot_shape
        )
        return True

    def store_batch(self, routing_data: np.ndarray, slot_mapping: np.ndarray) -> bool:
        """Write routing to mmap, or report that transport fallback is needed."""
        if not self._ensure_mmap_attached():
            return False
        assert self._array is not None, "shared routing mmap was not attached"
        self._array[slot_mapping] = routing_data
        return True

    def close(self) -> None:
        """Release the mapping. The manager owns the file."""
        self._array = None
        if self._mmap_obj is not None:
            try:
                self._mmap_obj.close()
            except Exception:
                logger.warning(
                    "Failed to close worker routed-experts mmap", exc_info=True
                )
            self._mmap_obj = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                logger.warning(
                    "Failed to close worker routed-experts file descriptor",
                    exc_info=True,
                )
            self._fd = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
