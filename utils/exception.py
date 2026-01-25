import sys


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information including
    file name, line number, and error message.
    """
    _, _, tb = error_detail.exc_info()

    # Extract file name and line number from traceback
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno

    # Construct detailed error message
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}] "
        f"with error message [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom Exception class for ML pipelines
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail
        )

    def __str__(self):
        return self.error_message
