def apply_bn(x, bn):
    return bn(x.transpose(1,2)).transpose(1,2).contiguous()