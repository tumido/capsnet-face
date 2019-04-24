import keras.backend as k

def margin_loss(true, predicted):
    """Margin loss.

    Args:
        true ([None, markers_count]): Label
        predicted ([None, capsules_count]): Predicted label

    Returns:
        Scalar loss
    """
    loss = true * k.square(k.maximum(.0, .9 - predicted)) + \
        .5 * (1 - true) * k.square(k.maximum(.0, predicted - .1))

    return k.mean(k.sum(loss, axis=1))
