"""Stable controlled failures at semantic boundaries."""


class ETVError(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class InputError(ETVError):
    pass


class UnsupportedSemantics(ETVError):
    pass


class ResourceLimit(ETVError):
    def __init__(self, detail: str) -> None:
        super().__init__("RESOURCE_LIMIT", detail)


class InternalError(ETVError):
    def __init__(self, detail: str) -> None:
        super().__init__("INTERNAL_ERROR", detail)
