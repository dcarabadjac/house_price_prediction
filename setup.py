from setuptools import setup, find_packages

setup(
    name="house_price_prediction",
    version="0.1.0",
    author="Denis Carabadjac",
    description="Predict the price of apartments in Chisinau based on various features",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "beautifulsoup4>=4.12",
        "geopy>=2.4",
        "pandas>=2.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.3",
        "selenium>=4.0",
        "tqdm>=4.0",
        "webdriver-manager>=4.0",
    ],
    python_requires=">=3.7",
)
