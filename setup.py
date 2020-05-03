from setuptools import find_packages, setup
setup(
    name='chordify_web',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask', 'werkzeug', 'sklearn', 'numpy', 'librosa', 'scipy', 'matplotlib', 'pandas', 'PyYAML', 'joblib',
        'lark-parser'
    ],
)