#!/bin/bash

echo Remove src/tensorRT/onnx_parser
rm -rf src/tensorRT/onnx_parser

echo Copy [onnx_parser/onnx_parser_8.6] to [src/tensorRT/onnx_parser]
cp -r onnx_parser/onnx_parser_8.6 src/tensorRT/onnx_parser

echo Configure your tensorRT path to 8.6
echo After that, you can execute the command 'make rmdetr -j64'