from models.Attention import cbam, se, eca, siam, simam, pfesa, identity
import sys


def get_attention_module2d(name, channels, **kwargs):
    if name == 'CBAM':
        return cbam.CBAM(channels)
    elif name == 'SE':
        return se.SEAttention(channels)
    elif name == 'ECA':
        return eca.ECA(channels)
    elif name == 'SIAM':
        return siam.SIAM2D()
    elif name == 'SimAM':
        return simam.SimAM2D()
    elif name == 'PFESA':
        return pfesa.PFESA()
    elif name == 'Identity':
        return identity.Identity()
    else:
        print("Attention module name is not valid.")
        sys.exit(1)


def get_attention_module_3d(name, channels):
    if name == 'CBAM':
        return cbam.CBAM3D(channels)
    elif name == 'SE':
        return se.SEAttention3D(channels)
    elif name == 'ECA':
        return eca.ECA3D(channels)
    elif name == 'SIAM':
        return siam.SIAM()
    elif name == 'SimAM':
        return simam.SimAM()
    elif name == 'PFESA':
        return pfesa.PFESA3D()
    elif name == 'Identity':
        return None
    else:
        print("Attention module name is not valid.")
        sys.exit(1)