from __future__ import annotations

from queue import Empty, Full, Queue
from typing import TypeVar

T = TypeVar("T")


def put_latest(queue_obj: Queue[T], item: T) -> None:
    try:
        queue_obj.put_nowait(item)
        return
    except Full:
        pass

    try:
        queue_obj.get_nowait()
    except Empty:
        pass

    try:
        queue_obj.put_nowait(item)
    except Full:
        pass
