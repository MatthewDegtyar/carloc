"""Register upsample_bicubic2d for coremltools by mapping it to bilinear.

It appears once, in DINOv2's positional-embedding interpolation, which resamples a
small fixed grid (~37x37) to the patch layout. At a fixed input resolution that
resample is the same every frame, so bicubic-vs-bilinear is a constant, tiny
difference on a learned embedding rather than anything on the image path. The
export is checked numerically against the PyTorch model afterwards.
"""
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op


@register_torch_op(torch_alias=["upsample_bicubic2d"], override=True)
def upsample_bicubic2d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    output_size = inputs[1]
    align_corners = bool(inputs[2].val) if len(inputs) > 2 and inputs[2].val is not None else False

    if output_size is not None and output_size.val is not None:
        size = [int(v) for v in output_size.val]
        out = mb.resize_bilinear(
            x=x, target_size_height=size[0], target_size_width=size[1],
            sampling_mode="ALIGN_CORNERS" if align_corners else "DEFAULT",
            name=node.name,
        )
    else:
        # PyTorch passes the scale factors either as two scalars or as one
        # length-2 array, depending on how F.interpolate was called. Assuming
        # scalars raises "only length-1 arrays can be converted to Python scalars".
        def _scalars(index):
            if len(inputs) <= index or inputs[index] is None or inputs[index].val is None:
                return None
            value = inputs[index].val
            try:
                return [float(v) for v in value]
            except TypeError:
                return [float(value)]

        scales = _scalars(3) or []
        if len(scales) < 2:
            scales += _scalars(4) or []
        if len(scales) < 2:
            scales = [scales[0] if scales else 1.0] * 2
        out = mb.upsample_bilinear(
            x=x, scale_factor_height=scales[0], scale_factor_width=scales[1],
            align_corners=align_corners, name=node.name,
        )
    context.add(out)
