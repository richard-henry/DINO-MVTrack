"""Model definitions for Track3D-TTO."""

def __getattr__(name):
    # encoders is also imported by nets.pips2 and by the isolated extractor.
    # Avoid an eager circular import and unnecessary tracking dependencies.
    if name in ('Pips', 'Track3DTTO'):
        from .track3d_tto import Pips, Track3DTTO
        return {'Pips': Pips, 'Track3DTTO': Track3DTTO}[name]
    raise AttributeError(name)

__all__ = ["Pips", "Track3DTTO"]
