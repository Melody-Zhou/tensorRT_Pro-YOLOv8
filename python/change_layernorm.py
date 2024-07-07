import onnx
import onnx_graphsurgeon as gs

# 加载 ONNX 模型
input_model_path = "rtmo-s_8xb32-600e_body7-640x640.onnx"
output_model_path = "rtmo-s_8xb32-600e_body7-640x640.plugin.onnx"
graph = gs.import_onnx(onnx.load(input_model_path))

# 遍历图中的所有节点
for node in graph.nodes:
    if node.op == "LayerNormalization":
        node.op = "CustomLayerNormalization"
        # 添加自定义属性
        node.attrs["name"] = "LayerNormPlugin"
        node.attrs["info"] = "This is custom LayerNormalization node"

# 删除无用的节点和张量
graph.cleanup()

# 导出修改后的模型
onnx.save(gs.export_onnx(graph), output_model_path)