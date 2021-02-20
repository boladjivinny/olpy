class NotFittedError(Exception):
    """Raised when an unfitted model is used for a prediction

    Attributes:
        previous (str): state at beginning of transition.
        next (str): attempted new state
        message (str): explanation of why the specific action 
            Failed.
    """

    def __init__(self, previous, next, message):
        self.previous = previous
        self.next = next
        self.message = message
