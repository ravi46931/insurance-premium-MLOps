import os
import sys
from colorama import Fore, Style

# from src.logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        f"\n{Style.BRIGHT + Fore.RED}Error occurred python script name: {Style.RESET_ALL}"
        + f"{Fore.YELLOW}[{file_name}]{Style.RESET_ALL}"
        f"\n{Style.BRIGHT + Fore.RED}Line number: {Style.RESET_ALL}"
        + f"{Fore.YELLOW}[{exc_tb.tb_lineno}]{Style.RESET_ALL}"
        f"\n{Style.BRIGHT + Fore.RED}Error message: {Style.RESET_ALL}"
        + f"{Fore.YELLOW}[{str(error)}]{Style.RESET_ALL}"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        # logging.info("HTH FYBHB HVBHVHV")
        s = 1 / "o"
    except Exception as e:
        raise CustomException(e, sys) from e
