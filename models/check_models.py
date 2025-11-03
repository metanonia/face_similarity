import onnx
print("=== Yolo11n Detection Model ===")
model = onnx.load("yolo11n.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")


print("=== Yolov8n_100e Detection Model ===")
model = onnx.load("yolov8n_100e.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")


print("\n=== ArcFace Model ===")
model = onnx.load("arcface.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")


print("\n=== Blaze Model ===")
model = onnx.load("blaze.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")


print("\n=== Face_detector (google) Model ===")
model = onnx.load("face_detector.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
