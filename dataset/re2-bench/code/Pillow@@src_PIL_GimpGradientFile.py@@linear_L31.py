def linear(middle, pos):
    if pos <= middle:
        if middle < EPSILON:
            return 0.0
        else:
            return 0.5 * pos / middle
    else:
        pos = pos - middle
        middle = 1.0 - middle
        if middle < EPSILON:
            return 1.0
        else:
            return 0.5 + 0.5 * pos / middle
