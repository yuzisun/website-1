## Deploy the LLaMA model with vLLM Runtime
Serving LLM models can be surprisingly slow even on high end GPUs, [vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use LLM inference engine. It can achieve 10x-20x higher throughput than Huggingface transformers.
It supports [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference) for increased throughput and GPU utilization,
[paged attention](https://vllm.ai) to address the memory bottleneck where in the autoregressive decoding process all the attention key value tensors(KV Cache) are kept in the GPU memory to generate next tokens.

You can deploy the LLaMA model with triton inference server vLLM backend using the `InferenceService` yaml API spec. 
Triton Inference Server vLLM backend implements the [Open Inference Protocol](https://github.com/kserve/open-inference-protocol) generate endpoint.

The Llama-2 7B AWQ model can be downloaded from [huggingface](https://huggingface.co/TheBloke/Llama-2-7B-AWQ/tree/main) and upload to your cloud storage.
AutoAWQ is an easy-to-use package for 4-bit quantized models. AutoAWQ speeds up models by 3x and reduces memory requirements by 3x compared to FP16.

### Upload the model to Cloud Storage
1. Create `config.pbtxt` with vLLM backend
```
backend: "vllm"

# Disabling batching in Triton, let vLLM handle the batching on its own.
max_batch_size: 0

# We need to use decoupled transaction policy for saturating
# vLLM engine for max throughtput.
# TODO [DLIS:5233]: Allow asynchronous execution to lift this
# restriction for cases there is exactly a single response to
# a single request.
model_transaction_policy {
  decoupled: True
}
# Note: The vLLM backend uses the following input and output names.
# Any change here needs to also be made in model.py
input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "stream"
    data_type: TYPE_BOOL
    dims: [ 1 ]
  },
  {
    name: "sampling_parameters"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

# The usage of device is deferred to the vLLM engine
instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]
```

2. Create a model repository with the following structure:

```
model_repository/
  vllm_model/
    config.pbtxt
    1/
      config.json
      model.safetensors
      model.json
      generation_config.json
```



### Deploy the model
=== "Yaml"
    ```yaml
    kubectl apply -n kserve-test -f - <<EOF
    apiVersion: serving.kserve.io/v1beta1
    kind: InferenceService
    metadata:
      name: llama-2-7b
    spec:
      predictor:
        model:
          modelFormat:
            name: huggingface
          runtime: kserve-tritonserver
          runtimeVersion: 23.10-vllm-python-py3
          storageUri: gs://kfserving-examples/models/triton/vllm/model_repository
          resources:
            limits:
              cpu: "4"
              memory: 50Gi
              nvidia.com/gpu: "1"
            requests:
              cpu: "1"
              memory: 50Gi
              nvidia.com/gpu: "1"
    ```

=== "kubectl"
```bash
kubectl get isvc llama-2-7b

```

## Benchmarking vLLM Runtime

You can download the benchmark testing data set by running
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

The tokenizer can be found from the downloaded llama model.

Now, assuming that your ingress can be accessed at
`${INGRESS_HOST}:${INGRESS_PORT}` or you can follow [this instruction](../../../../get_started/first_isvc.md#4-determine-the-ingress-ip-and-ports)
to find out your ingress IP and port.

You can run the [benchmarking script](./benchmark.py) and send the inference request to the exposed URL.

```bash
python benchmark.py --backend vllm --port ${INGRESS_PORT} --host ${INGRESS_HOST} --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer ./tokenizer --request-rate 5
```

!!! success "Expected Output"

    ```{ .json .no-copy }
       Total time: 216.81 s
       Throughput: 4.61 requests/s
       Average latency: 7.96 s
       Average latency per token: 0.02 s
       Average latency per output token: 0.04 s
    ```
