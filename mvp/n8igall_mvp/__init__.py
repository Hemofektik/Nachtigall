"""Nachtigall MVP: precipitation nowcasting (+60 min) from DWD RADOLAN radar data.

Pipeline: RADOLAN YW history + static DEM channels (+ later NWP wind)
-> compact U-Net -> RW target at +60 min.
"""

__version__ = "0.1.0"