# setup.py
# ------------------------------
# Package: qrretriever
# Author: wz1411@princeton.edu, xi.ye@princeton.edu
# Description: QRRetriever is a general purpose retriever using Query-focused Retrieval Heads
# ------------------------------

from setuptools import find_packages, setup


def get_requires():
    return [
        "torch",
        "transformers>=4.44.0",
        "flash_attn",
        "pyyaml>=5.1",
    ]

def get_console_scripts():
    return []

def main():
    setup(
        name="qrretriever",
        author="Wuwei Zhang, Xi Ye",
        author_email="wz1411@princeton.edu, xi.ye@princeton.edu",
        description="General purpose retriever using Query-focused Retrieval Heads",
        license="Apache 2.0 License",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.9.0",
        install_requires=get_requires(),
        package_data={
            "qrretriever": ["configs/*.yaml"],
        },
        include_package_data=True,
        entry_points={"console_scripts": get_console_scripts()},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
