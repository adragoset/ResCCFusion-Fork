FROM nvcr.io/nvidia/tensorrt:24.08-py3

COPY resccfusion_decoder.onnx /models/resccfusion_decoder.onnx
COPY resccfusion_encoder.onnx /models/resccfusion_encoder.onnx
COPY resccfusion_fusion.onnx /models/resccfusion_fusion.onnx

RUN /opt/tensorrt/python/python_setup.sh

COPY args_converter.py /models/converter/args_converter.py
COPY build_onnx_tensorrt.py /models/converter/build_onnx_tensorrt.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT [ "/start.sh" ]