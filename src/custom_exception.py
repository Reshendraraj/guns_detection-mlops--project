import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        """
        Custom exception class for capturing detailed error information.

        Args:
            error_message (str): A short description of the error.
            error_detail (sys): The sys module (used to extract traceback).
        """
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: sys) -> str:
        """
        Formats a detailed error message including file name and line number.

        Args:
            error_message (str): The base error message.
            error_detail (sys): The sys module for traceback access.

        Returns:
            str: A detailed formatted error message.
        """
        _, _, exc_tb = traceback.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in {file_name}, line {line_number}: {error_message}"
        return error_message

    def __str__(self):
        return self.error_message
