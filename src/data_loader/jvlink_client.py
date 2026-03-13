import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import win32com.client


@dataclass
class JVOpenResult:
    return_code: int
    read_count: Optional[int] = None
    download_count: Optional[int] = None
    last_file_timestamp: Optional[str] = None


@dataclass
class JVReadResult:
    return_code: int
    line: Optional[str] = None
    filename: Optional[str] = None
    buffer_size: Optional[int] = None


class JVLinkClient:
    def __init__(self, sid: str = "PythonJVLink", read_buffer_size: int = 110000):
        self.sid = sid
        self.read_buffer_size = read_buffer_size
        self.jv_link = None
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Initialize the JV-Link COM object."""
        try:
            try:
                self.jv_link = win32com.client.gencache.EnsureDispatch("JVDTLAB.JVLink")
            except Exception:
                self.jv_link = win32com.client.Dispatch("JVDTLAB.JVLink")

            ret = int(self.jv_link.JVInit(self.sid))
            if ret != 0:
                raise RuntimeError(f"JVInit failed with error code: {ret}")

            self.logger.info("JV-Link initialized successfully.")
        except Exception as exc:
            self.logger.error("Failed to initialize JV-Link: %s", exc)
            raise

    def set_save_path(self, save_path: str) -> int:
        """Set the JV-Link local data/cache directory."""
        self._ensure_initialized()
        result = int(self.jv_link.JVSetSavePath(save_path))
        if result != 0:
            raise RuntimeError(f"JVSetSavePath failed with error code: {result}")
        self.logger.info("JV-Link save path set to %s", save_path)
        return result

    def open_dataspec(self, dataspec: str, start_time: str, options: int = 1) -> JVOpenResult:
        """Open a JV-Link data stream and normalize the COM return shape."""
        self._ensure_initialized()

        raw_result = self._call_with_fallback(
            lambda: self.jv_link.JVOpen(dataspec, start_time, options),
            lambda: self.jv_link.JVOpen(dataspec, start_time, options, 0, 0, ""),
        )
        values = self._coerce_tuple(raw_result)

        result = JVOpenResult(
            return_code=self._coerce_int(values[0]),
            read_count=self._coerce_int(values[1]) if len(values) > 1 else None,
            download_count=self._coerce_int(values[2]) if len(values) > 2 else None,
            last_file_timestamp=self._coerce_text(values[3]) if len(values) > 3 else None,
        )
        if result.return_code < 0:
            raise RuntimeError(self._describe_jvopen_error(dataspec, result.return_code))

        self.logger.info(
            "JVOpen succeeded: dataspec=%s option=%s read_count=%s download_count=%s last_ts=%s",
            dataspec,
            options,
            result.read_count,
            result.download_count,
            result.last_file_timestamp,
        )
        return result

    def wait_for_download(self, expected_download_count: Optional[int], poll_interval: float = 1.0) -> int:
        """Wait until JVStatus reports that background downloads have completed."""
        self._ensure_initialized()

        if not expected_download_count or expected_download_count <= 0:
            return 0

        while True:
            status = self.status()
            if status < 0:
                raise RuntimeError(f"JVStatus failed with error code: {status}")
            if status >= expected_download_count:
                return status
            time.sleep(poll_interval)

    def status(self) -> int:
        self._ensure_initialized()
        return int(self.jv_link.JVStatus())

    def read(self) -> JVReadResult:
        """Read one record from the currently open JV-Link stream."""
        self._ensure_initialized()

        raw_result = self._call_with_fallback(
            lambda: self.jv_link.JVRead("", self.read_buffer_size, ""),
            lambda: self.jv_link.JVRead(),
        )
        values = self._coerce_tuple(raw_result)
        if not values:
            raise RuntimeError("JVRead returned no values.")

        read_result = JVReadResult(return_code=self._coerce_int(values[0]))
        if len(values) > 1:
            read_result.line = self._coerce_text(values[1])
        if len(values) > 2:
            if isinstance(values[2], int):
                read_result.buffer_size = int(values[2])
                if len(values) > 3:
                    read_result.filename = self._coerce_text(values[3])
            else:
                read_result.filename = self._coerce_text(values[2])
                if len(values) > 3 and isinstance(values[3], int):
                    read_result.buffer_size = int(values[3])

        return read_result

    def close(self) -> None:
        """Close the current JV-Link session."""
        if not self.jv_link:
            return

        try:
            self.jv_link.JVClose()
            self.logger.info("JV-Link session closed.")
        finally:
            self.jv_link = None

    def skip(self) -> None:
        self._ensure_initialized()
        self.jv_link.JVSkip()

    def cancel(self) -> None:
        self._ensure_initialized()
        self.jv_link.JVCancel()

    def _ensure_initialized(self) -> None:
        if self.jv_link is None:
            raise RuntimeError("JV-Link is not initialized.")

    @staticmethod
    def _coerce_tuple(result: Any) -> Tuple[Any, ...]:
        if isinstance(result, tuple):
            return result
        return (result,)

    @staticmethod
    def _coerce_int(value: Any) -> int:
        return int(value)

    @staticmethod
    def _coerce_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("cp932", errors="ignore").rstrip("\x00")
        return str(value).rstrip("\x00")

    @staticmethod
    def _call_with_fallback(*callables):
        last_error = None
        for callback in callables:
            try:
                return callback()
            except Exception as exc:
                last_error = exc
        raise last_error

    @staticmethod
    def _describe_jvopen_error(dataspec: str, code: int) -> str:
        if code == -112:
            return (
                f"JVOpen failed for {dataspec} with code -112 (JVERR_CONNECT). "
                "DataLab subscription/login/setup, session state, network connectivity, or FromTime may be invalid."
            )
        if code == -501:
            return (
                f"JVOpen failed for {dataspec} with code -501. "
                "Setup data mode could not proceed. Retry with option=3 so the setup-source dialog is shown again, "
                "or complete online initial setup from JV-Link settings first."
            )
        return f"JVOpen failed for {dataspec} with code: {code}"

