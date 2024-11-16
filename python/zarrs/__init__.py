from zarr.registry import register_pipeline

from .pipeline import ZarrsCodecPipeline as _ZarrsCodecPipeline
from .utils import CollapsedDimensionError, DiscontiguousArrayError
from ._internal import __version__

# Need to do this redirection so people can access the pipeline as `zarrs.ZarrsCodecPipeline` instead of `zarrs.pipeline.ZarrsCodecPipeline`
class ZarrsCodecPipeline(_ZarrsCodecPipeline):
    pass


register_pipeline(ZarrsCodecPipeline)

__version__: str = __version__

__all__ = ["ZarrsCodecPipeline", "DiscontiguousArrayError", "CollapsedDimensionError"]
