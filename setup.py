"""Tag wheels containing an executable as platform-specific, without a Python ABI."""

import json
import platform
import runpy
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

SHARED_DATA = (
    "default_config.toml",
    "models.yaml",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/README.md",
)


class BuildPy(build_py):
    def run(self):
        if not self.editable_mode:
            # Removed modules and native assets must not survive a previous build.
            package = Path(self.build_lib) / "muse_glimmer"
            if package.exists():
                shutil.rmtree(package)
        super().run()
        if not self.editable_mode:
            for name in SHARED_DATA:
                destination = Path(self.build_lib) / "muse_glimmer" / name
                self.mkpath(str(destination.parent))
                self.copy_file(str(Path(__file__).parent / name), str(destination))

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        for name in SHARED_DATA:
            path = str(Path(self.build_lib) / "muse_glimmer" / name)
            if path not in outputs:
                outputs.append(path)
        return outputs

    def get_output_mapping(self):
        return {
            **super().get_output_mapping(),
            **{
                str(Path(self.build_lib) / "muse_glimmer" / name): str(Path(__file__).parent / name)
                for name in SHARED_DATA
            },
        }


class SourceDist(sdist):
    def run(self):
        self.acquisition = runpy.run_path(str(Path(__file__).parent / "packaging/source.py"))
        self.source = self.acquisition["source_checkout"]()
        super().run()

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        destination = Path(base_dir) / "third-party/executorch"
        self.acquisition["copy_sources"](self.source, destination)
        # Keep the source archive below PyPI's size limit without dropping build inputs.
        for path in destination.rglob("*"):
            if path.is_file() and (
                "docs" in path.relative_to(destination).parts
                or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".svg", ".pdf"}
            ):
                path.unlink()
        (destination.parent / "executorch-source.json").write_text(
            json.dumps(self.acquisition["source_metadata"]()) + "\n"
        )


class MacWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        minimum = platform.mac_ver()[0].split(".")[0]
        return "py3", "none", f"macosx_{minimum}_0_arm64"


setup(cmdclass={"bdist_wheel": MacWheel, "build_py": BuildPy, "sdist": SourceDist})
