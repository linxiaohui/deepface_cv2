import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepface_cv2",
    version="0.0.0",
    author="Lin Xiao Hui",
    author_email="llinxiaohui@126.com",
    description="deepface with opencv, no need for tensorflow/keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/linxiaohui/deepface_cv2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    install_requires=["opencv-python"]
)
