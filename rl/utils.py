import tensorflow as tf

def infer_shape(x):
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.shape.dims is None:
        return tf.shape(x)

    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    ret = []
    for i in range(len(static_shape)):
        dim = static_shape[i]
        if dim is None:
            dim = dynamic_shape[i]
        ret.append(dim)

    return ret

def merge_last_two_dims(tensor):
    shape = infer_shape(tensor)
    shape[-2] *= shape[-1]
    shape.pop(-1)
    return tf.reshape(tensor, shape)