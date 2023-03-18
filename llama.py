#!/usr/bin/env python
"""
Driver script for Llama.cpp.
"""
import dataclasses
import enum
import os
from pathlib import Path

from typing import Self, NoReturn


class Model(enum.Enum):
    """Predefined relative paths to model files."""
    Q4_07 = Path("7B/ggml-model-q4_0.bin")
    Q4_13 = Path("13B/ggml-model-q4_0.bin")
    Q4_30 = Path("30B/ggml-model-q4_0.bin")
    Q4_65 = Path("65B/ggml-model-q4_0.bin")

    F16_07 = Path("7B/ggml-model-f16.bin")
    F16_13 = Path("13B/ggml-model-f16.bin")
    F16_30 = Path("30B/ggml-model-f16.bin")
    F16_65 = Path("65B/ggml-model-f16.bin")


@dataclasses.dataclass
class Prompt:
    """A prompt and optional reverse_prompt, specified as strings or paths."""

    prompt: Path | str
    """Prompt; when generating args:
    - if a string, adds "--prompt" argument. 
    - if a Path, adds "--file" argument. 
    """

    reverse_prompt: Path | str | None = None
    """Reverse prompt; when generating args: 
    - if None, not included.
    - if not None, adds the "--reverse-prompt" argument.
    - if a Path, the content is read into memory and used as a string.
    """

    def get_args(self) -> list[str]:
        """Get the prompt arguments as a list of strings."""
        if isinstance(self.prompt, Path):
            args = ["--file", str(self.prompt)]
        else:
            args = ["--prompt", self.prompt]

        if self.reverse_prompt is not None:
            if isinstance(self.reverse_prompt, Path):
                rp = self.reverse_prompt.read_text()
            else:
                rp = self.reverse_prompt
            args += ["--reverse-prompt", rp]

        return args

    def asdict(self) -> dict:
        """Convert this class to a serializable dictionary."""
        prompt_key = "path" if isinstance(self.prompt, Path) else "value"
        content = {prompt_key: str(self.prompt)}
        if self.reverse_prompt is not None:
            prompt_key = "path" if isinstance(self.reverse_prompt, Path) else "value"
            content["reverse_prompt"] = {prompt_key: str(self.reverse_prompt)}
        return content

    @classmethod
    def from_dict(cls, content: dict):
        """Create an instance from a serialized dictionary."""
        kwargs = {}
        if "value" in content:
            kwargs["prompt"] = content["value"]
        else:
            kwargs["prompt"] = Path(content["path"])

        if "reverse_prompt" in content:
            rp = content["reverse_prompt"]
            if "value" in rp:
                kwargs["reverse_prompt"] = rp["value"]
            else:
                kwargs["reverse_prompt"] = Path(rp["path"])

        return cls(**kwargs)


class ChatPrompt(enum.Enum):
    """Relative paths to predefined prompts and their corresponding reverse prompts."""
    Simple = Prompt(Path("chat-prompt.txt"), "User:")


@dataclasses.dataclass
class Llama:
    """Collection of argument parameters for Llama.cpp.

    Use this to generate argument lists or directly exec an instance.
    """
    prompt: Prompt
    models_dir: Path = Path("models")
    model: Model = Model.Q4_07
    threads: int = 4
    n_predict: int = 1024
    repeat_penalty: float = 1.17647
    repeat_last_n: int = 256
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.5
    context_size: int = 512
    batch_size: int = 16

    seed: int | None = None
    color: bool = False
    interactive: bool = False

    def as_chat(self) -> Self:
        """Return a new Llama instance with some parameters modified for "chat" use."""
        return dataclasses.replace(self, repeat_penalty=1.0, color=True, interactive=True)

    def get_args(self) -> list[str]:
        """Turn the current instance into a list of arguments."""
        args = [
            "llama",  # executable name
            "--model", str(self.models_dir / self.model.value),
            "--threads", f"{self.threads:d}",
            "--n_predict", f"{self.n_predict:d}",
            "--repeat_penalty", f"{self.repeat_penalty:f}",
            "--repeat_last_n", f"{self.repeat_last_n:d}",
            "--temp", f"{self.temperature:f}",
            "--top_k", f"{self.top_k:d}",
            "--top_p", f"{self.top_p:f}",
            "--ctx_size", f"{self.context_size:d}",
            "--batch_size", f"{self.batch_size:d}",
        ]

        if self.color:
            args += ["--color"]
        if self.interactive:
            args += ["--interactive"]
        if self.seed is not None:
            args += ["--seed", f"{self.seed:d}"]

        args += self.prompt.get_args()
        return args

    def launch(self, exec_path: Path = Path("main")) -> NoReturn:
        """Replace the current process by exec'ing the inference model."""
        args = self.get_args()
        args[0] = str(exec_path)
        os.execv(exec_path.resolve(), args)

    def asdict(self) -> dict:
        """Return a serializable dictionary from this instance."""
        content = dataclasses.asdict(self)
        content["prompt"] = self.prompt.asdict()
        content["models_dir"] = str(self.models_dir)
        content["model"] = self.model.name

        return content

    @classmethod
    def from_dict(cls, content: dict) -> Self:
        """Create an instance from a serialized version of this class."""
        kwargs = dict(content)
        try:
            kwargs["prompt"] = Prompt.from_dict(content["prompt"])
            kwargs["models_dir"] = Path(content["models_dir"])
            model_name = content["model"]
        except KeyError as e:
            raise ValueError("invalid dictionary content") from e

        try:
            kwargs["model"] = next(m for m in Model if m.name == model_name)
        except StopIteration:
            raise ValueError(f"unknown model name {model_name}")

        return cls(**kwargs)


def main():
    """Run this script."""
    import random
    import json
    rng = random.Random()

    base_dir = Path("../llama/")
    llama = Llama(
        prompt=ChatPrompt.Simple.value,
        # prompt=Prompt(Path("./story.txt")),
        models_dir=base_dir / "models",
        seed=rng.randint(0, 1 << 32 - 1),
    ).as_chat()

    as_json = json.dumps(llama.asdict(), indent=2)
    print(as_json)
    llama2 = Llama.from_dict(json.loads(as_json))
    assert llama2 == llama

    llama.launch(exec_path=base_dir / "main")

if __name__ == '__main__':
    main()

