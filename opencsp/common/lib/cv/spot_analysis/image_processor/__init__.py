from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractAggregateImageProcessor import (
    AbstractAggregateImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.AverageByGroupImageProcessor import (
    AverageByGroupImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.BcsLocatorImageProcessor import BcsLocatorImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ColorspaceImageProcessor import ColorspaceImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ConvolutionImageProcessor import ConvolutionImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.CroppingImageProcessor import CroppingImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.CustomSimpleImageProcessor import CustomSimpleImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.EchoImageProcessor import EchoImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ExposureDetectionImageProcessor import (
    ExposureDetectionImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.FalseColorImageProcessor import FalseColorImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.HotspotImageProcessor import HotspotImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.InpaintImageProcessor import InpaintImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.LogScaleImageProcessor import LogScaleImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.NullImageSubtractionImageProcessor import (
    NullImageSubtractionImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.PopulationStatisticsImageProcessor import (
    PopulationStatisticsImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.PowerpointImageProcessor import PowerpointImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.RewindImageProcessor import RewindImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.SaveToFileImageProcessor import SaveToFileImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.StabilizationImageProcessor import StabilizationImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.SupportingImagesCollectorImageProcessor import (
    SupportingImagesCollectorImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.TargetBoardLocatorImageProcessor import (
    TargetBoardLocatorImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.View3dImageProcessor import View3dImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ViewAnnotationsImageProcessor import (
    ViewAnnotationsImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.ViewCrossSectionImageProcessor import (
    ViewCrossSectionImageProcessor,
)

# Make these classes available when importing cv.spot_analysis.image_processor.*
__all__ = [
    'AbstractAggregateImageProcessor',
    'AbstractSpotAnalysisImageProcessor',
    'AbstractVisualizationImageProcessor',
    'AverageByGroupImageProcessor',
    'BcsLocatorImageProcessor',
    'ColorspaceImageProcessor',
    'ConvolutionImageProcessor',
    'CroppingImageProcessor',
    'CustomSimpleImageProcessor',
    'EchoImageProcessor',
    'ExposureDetectionImageProcessor',
    'FalseColorImageProcessor',
    'HotspotImageProcessor',
    'InpaintImageProcessor',
    'LogScaleImageProcessor',
    'NullImageSubtractionImageProcessor',
    'PopulationStatisticsImageProcessor',
    'PowerpointImageProcessor',
    'RewindImageProcessor',
    'SaveToFileImageProcessor',
    'StabilizationImageProcessor',
    'SupportingImagesCollectorImageProcessor',
    'TargetBoardLocatorImageProcessor',
    'View3dImageProcessor',
    'ViewAnnotationsImageProcessor',
    'ViewCrossSectionImageProcessor',
]
