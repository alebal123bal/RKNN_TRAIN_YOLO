"""
Fix the airockchip ultralytics_yolov8 RKNN-format ONNX so it produces
3 combined outputs (one per stride) matching the C++ postprocess layout:

    [1, h, w, nc + 4*reg_max]   (NHWC / channel-last)
    channels: [cls_logits(nc), dfl_logits(4*16)]

The original model exports 9 NCHW tensors:
    0: DFL   stride-8  [1,64,80,80]
    1: Sigmoid(cls) stride-8  [1, 1,80,80]   <- ReduceSum-named
    2: Clip(ReduceSum) stride-8  [1, 1,80,80]
    3-5: same for stride-16
    6-8: same for stride-32

We expose the raw class logits (Conv output before Sigmoid) and concat
with the DFL tensors, then transpose to NHWC.  The existing C++
process() function works unchanged with these 3 outputs.
"""

import sys
import onnx
from onnx import helper, TensorProto, shape_inference

def main():
    inp  = sys.argv[1] if len(sys.argv) > 1 else "rknn_files/yolov8/best_yolov8n.onnx"
    out  = sys.argv[2] if len(sys.argv) > 2 else "rknn_files/yolov8/best_yolov8n_fixed.onnx"

    model = onnx.load(inp)
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    # Build tensor-name → shape map
    shape_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if dims:
            shape_map[vi.name] = dims

    # Find DFL logit tensors in current graph outputs
    # Shape: [1, dfl_ch, h, w] with dfl_ch > 1 and h == w
    dfl_tensors = {}  # h -> tensor_name
    dfl_ch = None
    for out_info in graph.output:
        s = [d.dim_value for d in out_info.type.tensor_type.shape.dim]
        if len(s) == 4 and s[0] == 1 and s[1] > 1 and s[2] == s[3] and s[2] > 0:
            dfl_tensors[s[2]] = out_info.name
            dfl_ch = s[1]

    # Find raw class logit tensors = direct inputs to Sigmoid nodes
    # Shape: [1, nc, h, w] where h matches a DFL grid size (h == w)
    raw_cls = {}  # h -> tensor_name
    nc = None
    for node in graph.node:
        if node.op_type == 'Sigmoid' and node.input[0] in shape_map:
            s = shape_map[node.input[0]]
            if len(s) == 4 and s[0] == 1 and s[2] == s[3] and s[2] in dfl_tensors:
                raw_cls[s[2]] = node.input[0]
                nc = s[1]

    print("Raw class logit tensors:", raw_cls)
    print("DFL logit tensors:      ", dfl_tensors)

    assert len(raw_cls) == 3,     f"Expected 3 class logit tensors, got {len(raw_cls)}"
    assert len(dfl_tensors) == 3, f"Expected 3 DFL tensors, got {len(dfl_tensors)}"

    new_nodes     = []
    new_out_infos = []

    out_ch = nc + dfl_ch  # e.g. 1 + 64 = 65

    # Assign strides by grid size: largest h → stride 8, then 16, 32
    for stride, h in zip([8, 16, 32], sorted(dfl_tensors.keys(), reverse=True)):
        cat_name = f"cat_s{stride}"
        trp_name = f"output_s{stride}"

        # Concat class logit [1,nc,h,w] + DFL logit [1,dfl_ch,h,w] → [1,out_ch,h,w]
        new_nodes.append(helper.make_node(
            'Concat',
            inputs=[raw_cls[h], dfl_tensors[h]],
            outputs=[cat_name],
            axis=1,
        ))

        # Transpose NCHW→NHWC: [1,out_ch,h,w] → [1,h,w,out_ch]
        new_nodes.append(helper.make_node(
            'Transpose',
            inputs=[cat_name],
            outputs=[trp_name],
            perm=[0, 2, 3, 1],
        ))

        new_out_infos.append(
            helper.make_tensor_value_info(trp_name, TensorProto.FLOAT, [1, h, h, out_ch])
        )

    graph.node.extend(new_nodes)
    del graph.output[:]
    graph.output.extend(new_out_infos)

    onnx.save(model, out)
    print(f"\nSaved: {out}")
    print("New outputs (3 combined NHWC tensors):")
    for o in model.graph.output:
        s = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name}: {s}")

main()
