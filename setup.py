from setuptools import setup, find_packages

setup(
    name='house_price_prediction',
    version='1.0.0',
    description='House Price Prediction ML Pipeline',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
    ],
)
