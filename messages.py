def debug(message):
    """
        Verbosity for debug msg
    """
    print("\033[33m{:s}\033[0m".format(message))


def normal(message):
    """
        Verbosity for normal msg
    """
    print(message)


def success(message):
    """
        Verbosity for success msg
    """
    print("\033[32m{:s}\033[0m".format(message))


def verbose(message):
    """
        Verbosity for info msg
    """
    print("\033[38;5;247m{:s}\033[0m".format(message))


def error(message):
    """
        Verbosity for error msg
    """
    print("\033[31m{:s}\033[0m".format(message))
    
def title(message):
	"""
		Verbosity for title msg
	"""
	print("\033[1m{:s}\033[0m".format(message))