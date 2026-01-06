"""
qctc2csv.py

用途:
    将 qctc.data 文本文件转换为 CSV 文件 qctc.csv。

用法:
    1. 在包含 qctc.data 的目录中运行:
           python qctc2csv.py
    2. 以函数方式调用:
           from qctc2csv import convert_qctc_to_csv
           convert_qctc_to_csv("qctc.data", "qctc.csv")

说明:
    本脚本逐行读取输入文件, 使用逗号作为分隔符解析字段,
    并原样写入新的 CSV 文件, 不会改变任何数值或字符串内容。
"""

import csv
from pathlib import Path


def convert_qctc_to_csv(input_path: str = "qctc.data", output_path: str = "qctc.csv") -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:
            writer.writerow(row)


if __name__ == "__main__":
    convert_qctc_to_csv()

