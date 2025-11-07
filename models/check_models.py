import onnx

print("\n=== Face_detector (google) Model ===")
model = onnx.load("face_detector.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")


print("\n=== Landmark (Insightface) Model ===")
model = onnx.load("det_500m.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
for node in model.graph.node:
    if node.op_type in ["Conv", "MaxPool", "AveragePool"]:
        for attr in node.attribute:
            if attr.name == "strides":
                print(f"{node.name or node.op_type} - strides:", attr.ints)


print("\n=== Landmark (Retinaface) Model ===")
model = onnx.load("retinaface-resnet50.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
for node in model.graph.node:
    if node.op_type in ["Conv", "MaxPool", "AveragePool"]:
        for attr in node.attribute:
            if attr.name == "strides":
                print(f"{node.name or node.op_type} - strides:", attr.ints)


print("\n=== Recognition (Insightface) Model ===")
model = onnx.load("w600k_mbf.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
