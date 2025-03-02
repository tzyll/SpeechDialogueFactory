import vllm
from vllm.sampling_params import GuidedDecodingParams, SamplingParams
from openai import OpenAI
import torch
from transformers import AutoTokenizer, GenerationConfig
from pydantic import BaseModel
import tqdm
from utils.misc import validate_and_parse_json_output, post_process_output
import logging
import json

logger = logging.getLogger(__name__)


class LLM:
    def __init__(self, args):
        self.args = args
        self.inference_mode = args.inference_mode

        # We allow two types of inference modes: 'api' and 'vllm'
        if self.inference_mode == "api":
            self.api_key = args.api_key
            self.base_url = args.base_url
            self.is_openai = True if self.base_url is None else False
            self.model = args.model
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.inference_mode == "vllm":
            self.model, self.tokenizer, self.generation_config = self.load_model(args)
        else:
            raise ValueError(f"Invalid inference mode: {self.inference_mode}")

    def load_model(self, args):
        # This is only for loading the model in the 'vllm' mode
        model = vllm.LLM(
            model=args.model,
            tensor_parallel_size=torch.cuda.device_count(),
            distributed_executor_backend="ray",
            enable_prefix_caching=True,
            max_model_len=16384,
            gpu_memory_utilization=0.95,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        generation_config = GenerationConfig.from_pretrained(args.model)
        return model, tokenizer, generation_config

    def generate(self, prompts, json_model: BaseModel = None, **kwargs):
        if self.inference_mode == "api":
            return self.generate_api(prompts, json_model, **kwargs)
        elif self.inference_mode == "vllm":
            return self.generate_vllm(prompts, json_model, **kwargs)
        else:
            raise ValueError(f"Invalid inference mode: {self.inference_mode}")

    def generate_api(self, prompts, json_model: BaseModel = None):
        def generate_one_sample(prompt):
            if json_model is None:
                completion = self.client.beta.chat.completions.create(
                    model=self.model, messages=prompt
                )
                return completion.choices[0].message.content
            else:
                try:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=prompt,
                        response_format=json_model,
                        extra_body=dict(guided_decoding_backend="outlines"),
                    )
                    message = completion.choices[0].message
                    assert message.parsed
                    return message.parsed
                except Exception as e:
                    logger.error(f"Failed to parse JSON: {e}, {message}")
                    return None

        responses = []
        success_indices = []
        failed_indices = []
        for i, prompt in tqdm.tqdm(enumerate(prompts), total=len(prompts)):
            output = generate_one_sample(prompt)
            if output is not None:
                responses.append(output)
                success_indices.append(i)
            else:
                failed_indices.append(i)

        return {
            "responses": responses,
            "success_indices": success_indices,
            "failed_indices": failed_indices,
        }

    def generate_vllm(self, prompts, json_model: BaseModel = None, **kwargs):

        def setup_sampling_params(guided_decoding=None):
            # This function sets up the sampling parameters for the model
            # It prioritizes the parameters set given by the user, and falls back to default values (either from generation_config or hardcoded)
            temperature = kwargs.get(
                "temperature",
                (
                    0.7
                    if self.generation_config.temperature is None
                    else self.generation_config.temperature
                ),
            )
            top_p = kwargs.get(
                "top_p",
                (
                    0.9
                    if self.generation_config.top_p is None
                    else self.generation_config.top_p
                ),
            )
            top_k = kwargs.get(
                "top_k",
                (
                    50
                    if self.generation_config.top_k is None
                    else self.generation_config.top_k
                ),
            )
            max_tokens = kwargs.get("max_tokens", 16384)
            guided_decoding = kwargs.get("guided_decoding", guided_decoding)
            return SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                guided_decoding=guided_decoding,
            )

        def run_unguided_inference(prompts):
            sampling_params = setup_sampling_params()
            model_inputs = [
                self.tokenizer.apply_chat_template(prompt, tokenize=False)
                for prompt in prompts
            ]
            logger.info(f"Running unguided decoding with {len(model_inputs)} prompts")
            outputs = self.model.generate(model_inputs, sampling_params=sampling_params)
            outputs = [
                post_process_output(output.outputs[0].text) for output in outputs
            ]
            return outputs

        def run_guided_inference(prompts):
            json_schema = json_model.model_json_schema()
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
            sampling_params = setup_sampling_params(guided_decoding_params)
            model_inputs = [
                self.tokenizer.apply_chat_template(prompt, tokenize=False)
                for prompt in prompts
            ]
            logger.info(f"Running guided decoding with {len(model_inputs)} prompts")
            outputs = self.model.generate(model_inputs, sampling_params=sampling_params)
            outputs = [
                post_process_output(output.outputs[0].text) for output in outputs
            ]
            return outputs

        # For efficiency purpose, by default, we first run with unguided decoding
        # and then run with guided decoding if any samples are not valid JSON

        outputs = run_unguided_inference(prompts)
        assert len(outputs) == len(prompts)

        if json_model is None:
            return {
                "responses": outputs,
                "success_indices": list(range(len(prompts))),
                "failed_indices": [],
            }

        # Validate JSON outputs
        success_results = []
        failed_inputs = []
        for i, output in enumerate(outputs):
            result = validate_and_parse_json_output(output, json_model)
            if result is not None:
                success_results.append((i, result))
            else:
                failed_inputs.append((i, prompts[i]))

        if len(failed_inputs) > 0:
            logger.info(
                f"Failed to validate JSON for {len(failed_inputs)} samples. Running guided decoding."
            )
            guided_outputs = run_guided_inference(
                [prompt for _, prompt in failed_inputs]
            )
            assert len(guided_outputs) == len(failed_inputs)
            for (i, _), output in zip(failed_inputs, guided_outputs):
                result = validate_and_parse_json_output(output, json_model)
                if result is not None:
                    success_results.append((i, result))
                else:
                    logger.error(
                        f"Failed to validate JSON for guided decoding: {output} {result}"
                    )

        success_results.sort(key=lambda x: x[0])
        responses = [result for _, result in success_results]
        success_indices = [i for i, _ in success_results]
        failed_indices = [i for i in range(len(prompts)) if i not in success_indices]
        return {
            "responses": responses,
            "success_indices": success_indices,
            "failed_indices": failed_indices,
        }
