from ovito.modifiers import DislocationAnalysisModifier

try:
    modifier = DislocationAnalysisModifier()
    print("OVITO Pro 位错分析功能可用 ✅")
except ImportError:
    print("位错分析不可用，可能未安装 Pro 版本 ❌")