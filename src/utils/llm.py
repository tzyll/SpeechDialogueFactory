import vllm
import gc
from vllm.sampling_params import GuidedDecodingParams, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from openai import OpenAI, AzureOpenAI
import torch
from transformers import AutoTokenizer, GenerationConfig
from pydantic import BaseModel
import tqdm
from utils.misc import validate_and_parse_json_output, post_process_output
import logging
import json
import argparse
from utils.base_classes import SDFModule
import os
import contextlib

logger = logging.getLogger(__name__)

@SDFModule.set_role("generator")
class LLM(SDFModule):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--inference_mode",
            type=str,
            default="api",
            choices=["api", "vllm", "azure"],
            help="Inference mode to use: 'api' for OpenAI API, 'vllm' for VLLM.",
        )
        parser.add_argument(
            "--api_key",
            type=str,
            default=None,
            help="API key for OpenAI API. Required if inference_mode is 'api'.",
        )
        parser.add_argument(
            "--base_url",
            type=str,
            default=None,
            help="Base URL for OpenAI API. Required if inference_mode is 'api'.",
        )
        parser.add_argument(
            "--llm_in_use",
            type=str,
            default="meta-llama/Llama-3.3-70B-Instruct",
            help="Model name to use for inference.",
        )
        parser.add_argument(
            "--fast_mode",
            action="store_true",
            default=False,
            help="Use fast mode for inference. First use unguided decoding, then guided decoding if needed.",
        )

    def __init__(self, args):
        self.args = args
        self.inference_mode = args.inference_mode
        self.fast_mode = args.fast_mode

        # We allow two types of inference modes: 'api' and 'vllm'
        
    
    def initialize(self):
        if self.inference_mode == "api":
            self.api_key = self.args.api_key
            self.base_url = self.args.base_url
            self.model = self.args.llm_in_use
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.inference_mode == "azure":
            self.api_key = self.args.api_key
            self.base_url = self.args.base_url
            self.model = self.args.llm_in_use
            api_version = self.base_url.split("api-version=")[-1]
            self.client = AzureOpenAI(api_key=self.api_key, api_version=api_version, azure_endpoint=self.base_url)
        elif self.inference_mode == "vllm":
            self.model, self.tokenizer, self.generation_config = self.load_model(self.args)
        else:
            raise ValueError(f"Invalid inference mode: {self.inference_mode}")
        
        return self
    
    def unload(self):
        if self.inference_mode == "vllm":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            destroy_model_parallel()
            destroy_distributed_environment()
            del self.model.llm_engine.model_executor.driver_worker
            del self.model
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            import ray
            ray.shutdown()
        else:
            # For API, no explicit shutdown is needed
            pass


    def load_model(self, args):
        # This is only for loading the model in the 'vllm' mode
        model = vllm.LLM(
            model=args.llm_in_use,
            tensor_parallel_size=torch.cuda.device_count(),
            distributed_executor_backend="ray",
            enable_prefix_caching=True, 
            max_model_len=8192,
            # max_seq_len_to_capture=8192,
            gpu_memory_utilization=0.95,
            # max_num_seqs=16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.llm_in_use)
        generation_config = GenerationConfig.from_pretrained(args.llm_in_use)
        return model, tokenizer, generation_config

    def generate(self, prompts, json_model: BaseModel = None, **kwargs):
        if self.inference_mode == "api" or self.inference_mode == "azure":
            return self.generate_api(prompts, json_model, **kwargs)
        elif self.inference_mode == "vllm":
            return self.generate_vllm(prompts, json_model, **kwargs)
        else:
            raise ValueError(f"Invalid inference mode: {self.inference_mode}")

    def generate_api(self, prompts, json_model: BaseModel = None, **kwargs):
        def generate_one_sample(prompt):
            if json_model is None:
                completion = self.client.chat.completions.create(
                    model=self.model, messages=prompt, **kwargs
                )
                message = completion.choices[0].message.content
                return message
            else:
                if self.fast_mode:
                    completion = self.client.chat.completions.create(
                        model=self.model, messages=prompt, **kwargs
                    )
                    message = completion.choices[0].message.content
                    logger.info(
                        f"Running unguided decoding with output: {message}"
                    )
                    result = validate_and_parse_json_output(message, json_model)
                    if result is not None:
                        return result
                    logger.info(
                        f"Failed to validate JSON for unguided decoding, turning to guided decoding. {message}"
                    )
                try:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=prompt,
                        response_format=json_model,
                        extra_body=dict(guided_decoding_backend="outlines"),
                    )
                    message = completion.choices[0].message
                    logger.info(f"Running guided decoding with output: {message.parsed}")
                    assert message.parsed
                    return message.parsed.model_dump()
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
            repetition_penalty = kwargs.get(
                "repetition_penalty",
                (
                    1.05
                    if getattr(self.generation_config, "repetition_penalty", None) is None
                    else self.generation_config.repetition_penalty
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
                repetition_penalty=repetition_penalty
            )

        def run_unguided_inference(prompts):
            sampling_params = setup_sampling_params()
            model_inputs = [
                self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
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
                self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
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

        if json_model is None:
            outputs = run_unguided_inference(prompts)
            assert len(outputs) == len(prompts)
            return {
                "responses": outputs,
                "success_indices": list(range(len(prompts))),
                "failed_indices": [],
            }

        failed_inputs = [
            (i, prompt) for i, prompt in enumerate(prompts) if prompt is None
        ]
        success_results = []
        if self.fast_mode:
            outputs = run_unguided_inference(prompts)
            assert len(outputs) == len(prompts)

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
                    f"Failed to validate JSON for {len(failed_inputs)} samples. Will run guided decoding later."
                )

        guided_outputs = run_guided_inference([prompt for _, prompt in failed_inputs])
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
